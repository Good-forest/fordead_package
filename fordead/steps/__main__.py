#!/usr/bin/env python
# dest \= "(\w+)",
import click
import collections
from fordead.steps.step1_compute_masked_vegetationindex import cli_compute_masked_vegetationindex
from fordead.steps.step2_train_model import cli_train_model
from fordead.steps.step3_dieback_detection import cli_dieback_detection
from fordead.steps.step4_compute_forest_mask import cli_compute_forest_mask
from fordead.steps.step5_export_results import cli_export_results
from fordead.cli.cli_read_tileinfo import cli_read_tileinfo
from fordead.visualisation.vi_series_visualisation import cli_vi_series_visualisation
from fordead.visualisation.create_timelapse import cli_create_timelapse

class OrderedGroup(click.Group):
    # ref: https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
    def __init__(self, name=None, commands=None, **attrs):
        super(OrderedGroup, self).__init__(name, commands, **attrs)
        #: the registered subcommands by their exported names.
        self.commands = commands or collections.OrderedDict()

    def list_commands(self, ctx):
        return self.commands

@click.group(cls=OrderedGroup, context_settings={'help_option_names': ['-h', '--help']})
def fordead():
    """
    fordead - Remote sensing time series processing to detect forest anomalies

    The usual workflow is :
        masked_vi --> train_model --> dieback_detection --> forest_mask --> export_results
    """
fordead.add_command(cli_compute_masked_vegetationindex)
fordead.add_command(cli_train_model)
fordead.add_command(cli_dieback_detection)
fordead.add_command(cli_compute_forest_mask)
fordead.add_command(cli_export_results)
fordead.add_command(cli_read_tileinfo)
fordead.add_command(cli_vi_series_visualisation)
fordead.add_command(cli_create_timelapse)

if __name__ == '__main__':  # pragma: no cover
    fordead()