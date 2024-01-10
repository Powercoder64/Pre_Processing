import torch
from omegaconf import OmegaConf
import requests
from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check

def parallel_feature_extraction(args):
    process_name = "Pre_Processing"

    try:

        post_process_status(process_name, 'started')
        print(post_process_status(process_name , 'started').text)

        if args.feature_type == 'i3d':
            from models.i3d.extract_i3d import ExtractI3D  # defined here to avoid import errors
            extractor = ExtractI3D(args)
        elif args.feature_type == 'r21d':
            from models.r21d.extract_r21d import ExtractR21D  # defined here to avoid import errors
            extractor = ExtractR21D(args)
        elif args.feature_type == 'vggish':
            from models.vggish.extract_vggish import ExtractVGGish  # defined here to avoid import errors
            extractor = ExtractVGGish(args)
        elif args.feature_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            from models.resnet.extract_resnet import ExtractResNet
            extractor = ExtractResNet(args)
        elif args.feature_type == 'raft':
            from models.raft.extract_raft import ExtractRAFT
            extractor = ExtractRAFT(args)
        elif args.feature_type == 'pwc':
            from models.pwc.extract_pwc import ExtractPWC
            extractor = ExtractPWC(args)
        else:
            raise NotADirectoryError

        video_paths = form_list_from_user_input(args)
        indices = torch.arange(len(video_paths))
        replicas = torch.nn.parallel.replicate(extractor, args.device_ids[:len(indices)])
        inputs = torch.nn.parallel.scatter(indices, args.device_ids[:len(indices)])
        torch.nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)
        # closing the tqdm progress bar to avoid some unexpected errors due to multi-threading
        extractor.progress.close()
        post_process_status(process_name, 'completed')
        print(post_process_status('SampleProcess', 'completed').text)
    except Exception as e:

        error_message = str(e)

        post_process_status(process_name, 'error', error_message)
        print(post_process_status(process_name , 'error', error_message).text)


def post_process_status(process_name, status, error_message=''):
    # Placeholder for the real API URL

    message_api_url = "https://<real_rc_uva_url>"
    headers = {

        'API-Key': '<your_api_key_here>',
        'API-Secret': '<your_api_secret_here>'

    }

    # Setting up the payload with process details

    payload = {
        'processName': process_name,
        'processStatus': status,  # can be: 'started', 'completed', 'error'
        'errorMessage': error_message  # Informative string for errors, empty for start/stop

    }

    # Making the POST request

    response = requests.post(message_api_url, json=payload, headers=headers)

    return response

if __name__ == "__main__":
    cfg_cli = OmegaConf.from_cli()
    print(cfg_cli)
    cfg_yml = OmegaConf.load(build_cfg_path(cfg_cli.feature_type))
    # the latter arguments are prioritized
    cfg = OmegaConf.merge(cfg_yml, cfg_cli)
    # OmegaConf.set_readonly(cfg, True)
    print(OmegaConf.to_yaml(cfg))
    # some printing
    if cfg.on_extraction in ['save_numpy', 'save_pickle']:
        print(f'Saving features to {cfg.output_path}')
    if cfg.keep_tmp_files:
        print(f'Keeping temp files in {cfg.tmp_path}')

    sanity_check(cfg)
    parallel_feature_extraction(cfg)
