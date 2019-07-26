from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = ''
    settings.network_path = '/home/tangjiuqi097/vot/pytracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.results_path = '/home/tangjiuqi097/vot/pytracking/pytracking/tracking_results/'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.otb_path = '/home/tangjiuqi097/data/OTB/data'
    settings.vot_path = '/home/tangjiuqi097/data/vot2019/vot_short/sequences'
    settings.vot_path = '/home/tangjiuqi097/data/vot2019/VOT2018ST/sequences'
    settings.vot_path = '/home/tangjiuqi097/data/vot2019/vot_long/sequences'
    settings.vot_path = '/home/tangjiuqi097/data/vot2019/vot_rgbd/sequences'

    return settings

