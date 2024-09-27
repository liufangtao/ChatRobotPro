def make_policy(cfg):
    print(f"config={cfg}")
    if cfg.name == "octo":
        from .octo_policy import OctoPolicy

        return OctoPolicy(**cfg.args)
    elif cfg.name == "replay":
        from .replay_policy import ReplayPolicy

        return ReplayPolicy(**cfg.args)
    elif cfg.name == "act":
        from .act_policy import ActPolicy
        return ActPolicy(**cfg.args)
