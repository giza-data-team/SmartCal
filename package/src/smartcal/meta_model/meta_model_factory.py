from smartcal.meta_model.meta_model import MetaModel
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager

config_manager = ConfigurationManager()


class MetaModelFactory:
    """
    Factory class for creating MetaModel instances with configurable metric and top_n.
    """
    def create_model(self, metric: str = config_manager.metric, top_n: int = config_manager.k_recommendations):
        """
        Create a MetaModel instance with the specified metric and number of recommendations.

        :param metric: The calibration metric to use (default from config).
        :param top_n: Number of top models to return (default from config).
        :return: MetaModel instance.
        """
        return MetaModel(metric=metric, top_n=top_n) 