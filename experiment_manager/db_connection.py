import logging
import atexit
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


Base = declarative_base()
config = ConfigurationManager()

ssh_tunnel = None
def start_ssh_tunnel():
    global ssh_tunnel
    if config.ssh_enabled and ssh_tunnel is None:
        logging.info("SSH tunnel is enabled; starting tunnel to remote DB...")
        ssh_tunnel = SSHTunnelForwarder(
            (config.ssh_host, config.ssh_port),
            ssh_username=config.ssh_user,
            ssh_pkey=config.ssh_key_path,
            remote_bind_address=('localhost', config.remote_bind_port),
            local_bind_address=('localhost', config.local_bind_port)
        )
        ssh_tunnel.start()
        logging.info(f"SSH tunnel active on localhost:{config.local_bind_port}")

def _stop_ssh_tunnel():
    global ssh_tunnel
    if ssh_tunnel and ssh_tunnel.is_active:
        logging.info("Stopping SSH tunnel...")
        ssh_tunnel.stop()
        ssh_tunnel = None


# 1) Start tunnel if enabled
start_ssh_tunnel()

# 2) Build DB connection URL

if config.ssh_enabled:
    db_host = "localhost"
    db_port = config.local_bind_port
    logging.info(f"Connecting via SSH tunnel -> {db_host}:{db_port}")
else:
    db_host = config.db_host
    db_port = config.db_port
    logging.info(f"Connecting directly to DB -> {db_host}:{db_port}")

DATABASE_URL = f"postgresql://{config.db_user}:{config.db_password}@{db_host}:{db_port}/{config.db_name}"

_engine = create_engine(DATABASE_URL)
try:
    with _engine.connect() as conn:
        print("✅ Successfully connected to the database.")
except Exception as e:
    print("❌ Failed to connect:", e)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

# 3) As a fallback, ensure tunnel closes on normal interpreter exit
atexit.register(_stop_ssh_tunnel)
