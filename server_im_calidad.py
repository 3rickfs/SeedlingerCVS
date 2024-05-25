#!/usr/bin/env python3

from multiprocessing import Manager
from multiprocessing import Process
import time

import servidor_modbus_mp as server
import Control_IM_CALIDAD as control
from src.entorno_MODBUS import Config_server_db_brazos

if __name__ == '__main__':
    entorno=Config_server_db_brazos("Parametros/imagen_calidad.json")
    
    with Manager() as manager:
        data_ir=[*range(8)]
        data_hr=[*range(3)]

        shared_list_ir = manager.list()
        shared_list_ir.append(data_ir)
        
        shared_list_hr = manager.list()
        shared_list_hr.append(data_hr)    
       
        pr_control=Process(name='control',target=control.main,args=(shared_list_ir,shared_list_hr,False))
        pr_comunicacion=Process(name='comunicacion',target=server.run_main_async,args=(shared_list_ir,shared_list_hr,False,entorno))

        pr_comunicacion.start()
        time.sleep(0.1)
        pr_control.start()

        pr_comunicacion.join()
        pr_control.join()