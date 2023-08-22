sem_port = None
sem_ip = None

def set_sem_port(port):
    global sem_port
    sem_port = port

def set_sem_ip(ip):
    global sem_ip
    sem_ip = ip

def connect_sem():
    try:
        import serialem
    except ImportError:
        raise ImportError("Could not import serialem. Please install it with pip install serialem")
    
    if sem_port is None and sem_ip is None:
        return serialem
    
    serialem.ConnectToSEM(sem_port, sem_ip)
    return serialem