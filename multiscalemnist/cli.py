"""Command Line Interface"""
import logging
from typing import Optional

import click
import numpy as np
from yacs.config import CfgNode

from multiscalemnist.config import get_config
from multiscalemnist.generate import generate_set
from multiscalemnist.mnist import fetch_mnist


@click.group(help="MultiScaleMNIST")
@click.option("--config-file", default=None, help="path to config file", type=str)
@click.pass_context
def main(ctx: click.Context, config_file: Optional[str]):
    """Main group for subcommands."""
    ctx.ensure_object(CfgNode)
    config = get_config(config_file=config_file)
    logging.basicConfig(level=logging.INFO)
    ctx.obj = config


@main.command(help="Generate dataset")
@click.pass_obj
def generate(config):
    np.random.seed(config.SEED)
    data = fetch_mnist()
    generate_set(config, data)
