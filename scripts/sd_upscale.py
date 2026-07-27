import math

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, images, devices
from modules.processing import Processed
from modules.shared import opts, state


class Script(scripts.Script):
    def title(self):
        return "SD upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        info = gr.HTML("<p style=\"margin-bottom:0.75em\">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>")
        overlap = gr.Slider(minimum=0, maximum=256, step=16, label='Tile overlap', value=64, elem_id=self.elem_id("overlap"))
        scale_factor = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label='Scale Factor', value=2.0, elem_id=self.elem_id("scale_factor"))
        upscaler_index = gr.Radio(label='Upscaler', choices=[x.name for x in shared.sd_upscalers], value=shared.sd_upscalers[0].name, type="index", elem_id=self.elem_id("upscaler_index"))

        return [info, overlap, upscaler_index, scale_factor]

    def run(self, p, _, overlap, upscaler_index, scale_factor):
        if isinstance(upscaler_index, str):
            upscaler_index = [x.name.lower() for x in shared.sd_upscalers].index(upscaler_index.lower())
        processing.fix_seed(p)
        upscaler = shared.sd_upscalers[upscaler_index]

        p.extra_generation_params["SD upscale overlap"] = overlap
        p.extra_generation_params["SD upscale upscaler"] = upscaler.name
        p.extra_generation_params["SD upscale scale factor"] = scale_factor

        seed = p.seed

        init_img = p.init_images[0]
        init_img = images.flatten(init_img, opts.img2img_background_color)

        if upscaler.name != "None":
            img = upscaler.scaler.upscale(init_img, scale_factor, upscaler.data_path)
        else:
            img = init_img

        devices.torch_gc()

        grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=overlap)
        p.extra_generation_params["SD upscale final size"] = f"{grid.image_w}x{grid.image_h}"

        # split_grid() creates independent Pillow images. Neither complete source
        # image is needed after the tiles have been created.
        del img
        del init_img

        batch_size = p.batch_size
        upscale_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        tile_records = []
        source_tiles = []

        for _y, _h, row in grid.tiles:
            for tiledata in row:
                tile_records.append(tiledata)
                source_tiles.append(tiledata[2])

        batch_count = math.ceil(len(source_tiles) / batch_size)
        state.job_count = batch_count * upscale_count

        print(f"SD upscaling will process a total of {len(source_tiles)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)} per upscale in a total of {state.job_count} batches.")

        result_images = []
        result_infos = []
        result_seeds = []

        for n in range(upscale_count):
            start_seed = seed + n
            p.seed = start_seed
            iteration_info = None
            iteration_complete = True

            for i in range(batch_count):
                if state.interrupted or state.stopping_generation:
                    iteration_complete = False
                    break

                batch_start = i * batch_size
                batch_end = min(batch_start + batch_size, len(source_tiles))
                current_batch = source_tiles[batch_start:batch_end]

                # Use the actual number of inputs in this slice. This is normally
                # one for SD Upscale, and avoids duplicating a short final slice.
                p.batch_size = len(current_batch)
                p.init_images = current_batch

                state.job = f"Batch {i + 1 + n * batch_count} out of {state.job_count}"
                batch_processed = processing.process_images(p)

                if iteration_info is None:
                    iteration_info = batch_processed.info

                if state.interrupted or state.stopping_generation:
                    iteration_complete = False
                    break

                expected_results = len(current_batch)
                batch_results = batch_processed.images[:expected_results]
                if len(batch_results) != expected_results:
                    iteration_complete = False
                    break

                p.seed = batch_processed.seed + expected_results

                # Put completed tiles directly into the output grid. On the last
                # iteration, also release each original crop after its final use.
                for offset, result in enumerate(batch_results):
                    tile_index = batch_start + offset
                    tile_records[tile_index][2] = result
                    if n == upscale_count - 1:
                        source_tiles[tile_index] = None

                del batch_processed

            if not iteration_complete:
                break

            combined_image = images.combine_grid(grid)
            result_images.append(combined_image)
            result_infos.append(iteration_info)
            result_seeds.append(start_seed)

            if opts.samples_save:
                images.save_image(combined_image, p.outpath_samples, "", start_seed, p.prompt, opts.samples_format, info=iteration_info, p=p)

        initial_info = result_infos[0] if result_infos else None
        processed = Processed(
            p,
            result_images,
            seed,
            initial_info,
            all_seeds=result_seeds or [seed],
            infotexts=result_infos or None,
        )

        return processed
