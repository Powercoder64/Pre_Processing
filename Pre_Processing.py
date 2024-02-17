import torch
from omegaconf import OmegaConf
import requests
import json
import sys
import os
from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check

def parallel_feature_extraction(args):
    request_type = "Pre_Processing"

    try:

        from models.i3d.extract_i3d import ExtractI3D  # defined here to avoid import errors
        extractor = ExtractI3D(args)

        video_paths = form_list_from_user_input(args)
        indices = torch.arange(len(video_paths))
        replicas = torch.nn.parallel.replicate(extractor, args.device_ids[:len(indices)])
        inputs = torch.nn.parallel.scatter(indices, args.device_ids[:len(indices)])
        torch.nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)
        # closing the tqdm progress bar to avoid some unexpected errors due to multi-threading
        extractor.progress.close()
        filename=args.filename
        messageid = args.messageid
        send_status_update(messageid, filename, request_type, 'preprocessing-completed', '...')

    except Exception as e:
        filename=args.filename
        messageid = args.messageid

        error_message = str(e)

        send_status_update(messageid, filename, request_type,'error', error_message)



def send_status_update(messageid, filename, request_type, response_type, comment=""):
    url = "http://aiai-service-service.aiai-ml-curvex-dev.svc.cluster.local/aiai/api/model_run_status_update"
    payload = json.dumps({
        "messageid": messageid,
        "filename": filename,
        "requestType": request_type,
        "responseType": response_type,
        "comment": comment
    })
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=payload)
    print(response.text)

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

    #sanity_check(cfg)
    parallel_feature_extraction(cfg)
