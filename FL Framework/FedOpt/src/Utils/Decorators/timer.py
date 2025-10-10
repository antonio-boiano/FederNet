import time
import logging

logger = logging.getLogger("FedOpt")

'''
Quando una funzione viene decorata con @normal_timer, il decoratore viene eseguito automaticamente 
quando la funzione viene definita. Non è necessario chiamare esplicitamente la funzione decoratrice. 
Il decoratore avvolge la funzione originale in un wrapper che misura il tempo di esecuzione e ne stampa il risultato. 
'''


def thread_timer(function):
    async def wrapper(*args, **kwargs):
        t1 = time.time()
        value = await function(*args, **kwargs)
        t2 = time.time()
        fname = function.__name__
        logger.info(f"[TIMER] {fname} -> {t2 - t1:.4f} sec.")
        return value

    return wrapper


def normal_timer(function):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        value = function(*args, **kwargs)
        t2 = time.time()
        fname = function.__name__
        logger.info(f"[TIMER] {fname} -> {t2 - t1:.4f} sec.")
        return value

    return wrapper
