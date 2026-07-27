import torch
import gradio as gr
import math

from backend.sampling.sampling_function import calc_cond_uncond_batch
from backend import attention, memory_management
from torch import einsum
from einops import rearrange, repeat
from modules import scripts, shared
from modules.ui_components import InputAccordion


attn_precision = memory_management.force_upcast_attention_dtype()


# Gaussian kernels are tiny, but rebuilding and transferring one every sampling
# step is unnecessary. Cache them by device, dtype, channel count and settings.
_gaussian_kernel_cache = {}


def attention_basic_with_sim(q, k, v, heads, mask=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head ** -0.5

    h = heads
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    # force cast to fp32 to avoid overflowing
    if attn_precision == torch.float32:
        sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return (out, sim)


def create_blur_map(x0, attn_scores, sigma=3.0, threshold=1.0):
    """Create SAG's selectively blurred latent from reduced attention scores.

    attn_scores is already reduced from [batch * heads, queries, keys] to
    [batch, keys]. Keeping only this score map avoids retaining the full
    quadratic attention tensor until the extra SAG UNet pass.
    """
    b, _, lh, lw = x0.shape
    hw = attn_scores.shape[-1]
    mask = attn_scores > threshold

    # Original sizing method. It works for normal inputs without Kohya HRFix
    # scaling; the fallback below handles the known HRFix shape override.
    ratio = 2**(math.ceil(math.sqrt(lh * lw / hw)) - 1).bit_length()
    h = math.ceil(lh / ratio)
    w = math.ceil(lw / ratio)

    if h * w != mask.size(1):
        kohya_shrink_shape = getattr(shared, 'kohya_shrink_shape', None)
        if kohya_shrink_shape:
            w = kohya_shrink_shape[0]
            h = kohya_shrink_shape[1]

    if h * w != mask.size(1):
        raise RuntimeError(
            f"SAG attention mask has {mask.size(1)} positions, but the "
            f"calculated spatial shape is {h}x{w}."
        )

    # The mask is binary and nearest-neighbor upsampling is used by default, so
    # keeping it in the latent dtype avoids promoting the degraded latent to
    # FP32 when attention itself was upcast.
    mask = mask.reshape(b, h, w).unsqueeze(1).to(device=x0.device, dtype=x0.dtype)
    mask = torch.nn.functional.interpolate(mask, (lh, lw), mode="nearest")

    blurred = gaussian_blur_2d(x0, kernel_size=9, sigma=sigma)
    return blurred * mask + x0 * (1 - mask)


def gaussian_blur_2d(img, kernel_size, sigma):
    # Sigma zero means no degradation. The caller normally bypasses SAG
    # entirely in this case, but keep this guard for direct calls as well.
    if sigma <= 0:
        return img

    channels = img.shape[-3]
    device_index = img.device.index if img.device.index is not None else -1
    cache_key = (
        img.device.type,
        device_index,
        img.dtype,
        channels,
        kernel_size,
        float(sigma),
    )
    kernel2d = _gaussian_kernel_cache.get(cache_key)

    if kernel2d is None:
        ksize_half = (kernel_size - 1) * 0.5

        # Match the original implementation: construct in default FP32, then
        # move/cast once. Subsequent sampling steps reuse the cached tensor.
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        x_kernel = (pdf / pdf.sum()).to(device=img.device, dtype=img.dtype)

        kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
        kernel2d = kernel2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
        _gaussian_kernel_cache[cache_key] = kernel2d

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]
    img = torch.nn.functional.pad(img, padding, mode="reflect")
    return torch.nn.functional.conv2d(img, kernel2d, groups=channels)


class SelfAttentionGuidance:
    def patch(self, model, scale, blur_sigma, threshold):
        m = model.clone()

        attn_scores = None

        # TODO: make this work properly with chunked batches
        #       currently, we can only save the attn from one UNet call
        def attn_and_record(q, k, v, extra_options):
            nonlocal attn_scores
            # if uncond, save the attention scores
            heads = extra_options["n_heads"]
            cond_or_uncond = extra_options["cond_or_uncond"]
            b = q.shape[0] // len(cond_or_uncond)
            if 1 in cond_or_uncond:
                uncond_index = cond_or_uncond.index(1)
                # Do the attention operation, then immediately reduce the
                # unconditional matrix to the only values SAG later needs:
                # mean across heads and sum across query positions.
                (out, sim) = attention_basic_with_sim(q, k, v, heads=heads)
                # When using a higher batch size, the expected result batch
                # dimension is [uc1, ... ucn, c1, ... cn].
                n_slices = heads * b
                start = n_slices * uncond_index
                end = n_slices * (uncond_index + 1)
                uncond_sim = sim[start:end]
                _, query_tokens, key_tokens = uncond_sim.shape
                attn_scores = (
                    uncond_sim
                    .reshape(b, heads, query_tokens, key_tokens)
                    .mean(dim=1)
                    .sum(dim=1)
                    .contiguous()
                )
                del uncond_sim, sim
                return out
            else:
                return attention.attention_function(q, k, v, heads=heads)

        def post_cfg_function(args):
            nonlocal attn_scores
            uncond_attn = attn_scores
            # Do not let a stale map survive into a later call, and make the
            # reduced map eligible for release before the extra UNet pass.
            attn_scores = None

            sag_scale = scale
            sag_sigma = blur_sigma
            sag_threshold = threshold
            model = args["model"]
            uncond_pred = args["uncond_denoised"]
            uncond = args["uncond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            x = args["input"]
            if min(cfg_result.shape[2:]) <= 4:  # skip when too small to add padding
                return cfg_result
            if uncond_attn is None:
                # The original implementation would fail here. Returning the
                # normal CFG result is safer for an unsupported chunk layout.
                return cfg_result

            # Create the adversarially blurred image. Release the attention
            # score map before the additional UNet/ControlNet evaluation.
            degraded = create_blur_map(uncond_pred, uncond_attn, sag_sigma, sag_threshold)
            del uncond_attn
            degraded_noised = degraded + x - uncond_pred
            # call into the UNet
            (sag, _) = calc_cond_uncond_batch(model, uncond, None, degraded_noised, sigma, model_options)
            return cfg_result + (degraded - sag) * sag_scale

        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)

        # from diffusers:
        # unet.mid_block.attentions[0].transformer_blocks[0].attn1.patch
        m.set_model_attn1_replace(attn_and_record, "middle", 0, 0)

        return (m,)


opSelfAttentionGuidance = SelfAttentionGuidance()


class SAGForForge(scripts.Script):
    sorting_priority = 12.5

    def title(self):
        return "SelfAttentionGuidance Integrated (SD 1.x, SD 2.x, SDXL)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            scale = gr.Slider(label='Scale', minimum=-2.0, maximum=5.0, step=0.01, value=0.5)
            blur_sigma = gr.Slider(label='Blur Sigma', minimum=0.0, maximum=10.0, step=0.01, value=2.0)
            threshold = gr.Slider(label='Blur mask threshold', minimum=0.0, maximum=4.0, step=0.01, value=1.0)

        self.infotext_fields = [
            (enabled, lambda d: d.get("sag_enabled", False)),
            (scale,         "sag_scale"),
            (blur_sigma,    "sag_blur_sigma"),
            (threshold,     "sag_threshold"),
        ]

        return enabled, scale, blur_sigma, threshold

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, scale, blur_sigma, threshold = script_args

        if not enabled:
            return

        # Scale zero produces no correction, and sigma zero produces no
        # degradation. Avoid installing the patch and running an otherwise
        # wasted extra UNet/ControlNet pass in either case.
        if math.isclose(scale, 0.0, abs_tol=1e-12) or blur_sigma <= 0:
            p.extra_generation_params.update(dict(
                sag_enabled     = enabled,
                sag_scale       = scale,
                sag_blur_sigma  = blur_sigma,
                sag_threshold   = threshold,
            ))
            return

        #   not for FLux
        if not shared.sd_model.is_webui_legacy_model():     #   ideally would be is_flux
            print("Self Attention Guidance is not compatible with Flux")
            return
        #   Self Attention Guidance errors if CFG is 1
        if p.is_hr_pass == False and p.cfg_scale <= 1:
            print("Self Attention Guidance requires CFG > 1")
            return
        if p.is_hr_pass == True and p.hr_cfg <= 1:
            print("Self Attention Guidance (hires pass) requires Hires CFG > 1")
            return

        unet = p.sd_model.forge_objects.unet

        unet = opSelfAttentionGuidance.patch(unet, scale, blur_sigma, threshold)[0]

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            sag_enabled     = enabled,
            sag_scale       = scale,
            sag_blur_sigma  = blur_sigma,
            sag_threshold   = threshold,
        ))

        return
