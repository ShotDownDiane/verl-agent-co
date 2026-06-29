__all__ = ["OsmV2AtomicEnv"]


def __getattr__(name):
    if name == "OsmV2AtomicEnv":
        from roll.pipeline.agentic.env.osm_v2.env import OsmV2AtomicEnv

        return OsmV2AtomicEnv
    raise AttributeError(name)
