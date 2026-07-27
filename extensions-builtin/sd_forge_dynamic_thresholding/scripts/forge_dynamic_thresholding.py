import gradio as gr

from modules import scripts
from lib_dynamic_thresholding.dynthres import DynamicThresholdingNode

opDynamicThresholdingNode = DynamicThresholdingNode().patch

SCHEDULER_CHOICES = [
    'Constant', 'Linear Down', 'Cosine Down', 'Half Cosine Down', 'Linear Up',
    'Cosine Up', 'Half Cosine Up', 'Power Up', 'Power Down', 'Linear Repeating',
    'Cosine Repeating', 'Sawtooth'
]

SETTINGS_ARGUMENT_COUNT = 12


class DynamicThresholdingForForge(scripts.Script):
    sorting_priority = 11

    def title(self):
        return "DynamicThresholding (CFG-Fix) Integrated"

    def show(self, is_img2img):
        # Make this extension visible in both txt2img and img2img tabs.
        return scripts.AlwaysVisible

    @staticmethod
    def create_settings_controls(label_prefix='', enabled_value=False):
        enabled = gr.Checkbox(label=f'{label_prefix}Enabled', value=enabled_value)
        mimic_scale = gr.Slider(label=f'{label_prefix}Mimic Scale', minimum=0.0, maximum=100.0, step=0.5,
                                value=7.0)
        threshold_percentile = gr.Slider(label=f'{label_prefix}Threshold Percentile', minimum=0.0, maximum=1.0,
                                         step=0.01, value=1.0)
        mimic_mode = gr.Radio(label=f'{label_prefix}Mimic Mode', choices=SCHEDULER_CHOICES, value='Constant')
        mimic_scale_min = gr.Slider(label=f'{label_prefix}Mimic Scale Min', minimum=0.0, maximum=100.0, step=0.5,
                                    value=0.0)
        cfg_mode = gr.Radio(label=f'{label_prefix}Cfg Mode', choices=SCHEDULER_CHOICES, value='Constant')
        cfg_scale_min = gr.Slider(label=f'{label_prefix}Cfg Scale Min', minimum=0.0, maximum=100.0, step=0.5,
                                  value=0.0)
        sched_val = gr.Slider(label=f'{label_prefix}Sched Val', minimum=0.0, maximum=100.0, step=0.01, value=1.0)
        separate_feature_channels = gr.Radio(label=f'{label_prefix}Separate Feature Channels',
                                             choices=['enable', 'disable'], value='enable')
        scaling_startpoint = gr.Radio(label=f'{label_prefix}Scaling Startpoint', choices=['MEAN', 'ZERO'],
                                      value='MEAN')
        variability_measure = gr.Radio(label=f'{label_prefix}Variability Measure', choices=['AD', 'STD'], value='AD')
        interpolate_phi = gr.Slider(label=f'{label_prefix}Interpolate Phi', minimum=0.0, maximum=1.0, step=0.01,
                                    value=1.0)

        return (
            enabled,
            mimic_scale,
            threshold_percentile,
            mimic_mode,
            mimic_scale_min,
            cfg_mode,
            cfg_scale_min,
            sched_val,
            separate_feature_channels,
            scaling_startpoint,
            variability_measure,
            interpolate_phi,
        )

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown('### Standard Pass Settings')
            gr.Markdown('Used for txt2img base generation, img2img, inpaint, SD Upscale, and other non-hires passes.')
            standard_settings = self.create_settings_controls()

            use_separate_hires_settings = gr.Checkbox(
                label='Use separate settings for the Hires.fix pass',
                value=False,
            )

            gr.Markdown('### Hires.fix Pass Settings')
            gr.Markdown('Used only for the second Hires.fix sampling pass when the checkbox above is enabled.')
            hires_settings = self.create_settings_controls(label_prefix='Hires.fix ', enabled_value=True)

        return (*standard_settings, use_separate_hires_settings, *hires_settings)

    @staticmethod
    def unpack_settings(settings):
        return dict(zip((
            'enabled',
            'mimic_scale',
            'threshold_percentile',
            'mimic_mode',
            'mimic_scale_min',
            'cfg_mode',
            'cfg_scale_min',
            'sched_val',
            'separate_feature_channels',
            'scaling_startpoint',
            'variability_measure',
            'interpolate_phi',
        ), settings))

    @staticmethod
    def add_generation_params(p, settings, prefix='dynthres_'):
        p.extra_generation_params.update({
            f'{prefix}enabled': settings['enabled'],
            f'{prefix}mimic_scale': settings['mimic_scale'],
            f'{prefix}threshold_percentile': settings['threshold_percentile'],
            f'{prefix}mimic_mode': settings['mimic_mode'],
            f'{prefix}mimic_scale_min': settings['mimic_scale_min'],
            f'{prefix}cfg_mode': settings['cfg_mode'],
            f'{prefix}cfg_scale_min': settings['cfg_scale_min'],
            f'{prefix}sched_val': settings['sched_val'],
            f'{prefix}separate_feature_channels': settings['separate_feature_channels'],
            f'{prefix}scaling_startpoint': settings['scaling_startpoint'],
            f'{prefix}variability_measure': settings['variability_measure'],
            f'{prefix}interpolate_phi': settings['interpolate_phi'],
        })

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This is called before every sampling pass. With Hires.fix it is called
        # for the base pass and again while p.is_hr_pass is True for the HR pass.
        if len(script_args) == SETTINGS_ARGUMENT_COUNT:
            # Backward compatibility for old saved/API configurations that only
            # provide the original standard-pass settings.
            standard_args = script_args
            use_separate_hires_settings = False
            hires_args = script_args
        elif len(script_args) == SETTINGS_ARGUMENT_COUNT * 2 + 1:
            standard_args = script_args[:SETTINGS_ARGUMENT_COUNT]
            use_separate_hires_settings = bool(script_args[SETTINGS_ARGUMENT_COUNT])
            hires_args = script_args[SETTINGS_ARGUMENT_COUNT + 1:]
        else:
            raise ValueError(
                'Dynamic Thresholding received an unexpected number of script arguments: '
                f'{len(script_args)}'
            )

        is_hires_pass = bool(getattr(p, 'is_hr_pass', False))
        use_hires_configuration = is_hires_pass and use_separate_hires_settings
        selected_args = hires_args if use_hires_configuration else standard_args
        settings = self.unpack_settings(selected_args)

        # Normal img2img/inpaint/SD Upscale never sets is_hr_pass, so the hires
        # configuration is ignored in those workflows.
        if use_hires_configuration:
            p.extra_generation_params['dynthres_hires_separate'] = True
            metadata_prefix = 'dynthres_hr_'
        else:
            metadata_prefix = 'dynthres_'

        if not settings['enabled']:
            # Record an explicitly disabled HR override so the infotext explains
            # why DT ran in the base pass but not in Hires.fix.
            if use_hires_configuration:
                p.extra_generation_params[f'{metadata_prefix}enabled'] = False
            return

        unet = p.sd_model.forge_objects.unet

        unet = opDynamicThresholdingNode(
            unet,
            settings['mimic_scale'],
            settings['threshold_percentile'],
            settings['mimic_mode'],
            settings['mimic_scale_min'],
            settings['cfg_mode'],
            settings['cfg_scale_min'],
            settings['sched_val'],
            settings['separate_feature_channels'],
            settings['scaling_startpoint'],
            settings['variability_measure'],
            settings['interpolate_phi'],
        )[0]

        p.sd_model.forge_objects.unet = unet

        # These values are written to the output infotext and do not influence
        # generation. Standard keys remain unchanged for compatibility; HR keys
        # use a separate dynthres_hr_ prefix.
        self.add_generation_params(p, settings, metadata_prefix)

        return
