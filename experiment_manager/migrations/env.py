from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Package.src.SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager
from experiment_manager.db_connection import start_ssh_tunnel
from experiment_manager.models import Base

config_manager = ConfigurationManager()

if config_manager.ssh_enabled:
    start_ssh_tunnel()
    
    db_host = "127.0.0.1"
    db_port = config_manager.local_bind_port
else:
    db_host = config_manager.db_host
    db_port = config_manager.db_port


# Create db URL from config manager credentails
DATABASE_URL = f"postgresql://{config_manager.db_user}:{config_manager.db_password}@{db_host}:{db_port}/{config_manager.db_name}"

# Set the SQLAlchemy database URL in the Alembic configuration
config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL)


# save config in a file if logging file is available
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


target_metadata = Base.metadata



def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()