def clean_session():
    import os
    import os.path
    import subprocess

    PATH_MN = os.path.expanduser("")

    exec_script = PATH_MN + ' -c'
    if os.path.exists(PATH_MN):
        subprocess.run(exec_script, shell=True)
    else:
        os.system("mn -c")

    from mininet.net import Containernet
    from mininet.node import Controller
    from mininet.cli import CLI
    from mininet.link import TCLink
    from mininet.log import info, setLogLevel
    setLogLevel('info')

    import docker
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Instantiate a Docker client
    client = docker.from_env()

    # Stop all containers
    containers = client.containers.list()
    for container in containers:
        container.stop()

    # Prune all containers
    client.containers.prune()

if __name__ == "__main__":
    clean_session()
