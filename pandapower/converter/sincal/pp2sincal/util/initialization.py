try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

try:
    import win32com.client
except:
    logger.warning(r'seems like you are not on a windows machine')
import time
import os
from glob import glob
import shutil


def create_simulation_environment(output_folder=None, file_name=None, use_active_net=True,
                                  use_ui=True, sincal_interaction=True, delete_files=True):
    '''
    This function creates the simulation environment and returns a tuple with the three main components
    for the conversion. The simulation Object (COM-Component), the sincal document and the sincal
    application object will be returned.

    :param output_folder: Path to the used folder
    :type output_folder: string
    :param file_name: Project name
    :type file_name: string
    :param use_active_net: Flag for the usage of the active network object in sincal.
    :type use_active_net: boolean
    :param use_ui: Use the open user interface
    :type use_ui: boolean
    :param sincal_interaction: Flag if Sincal interaction is wished
    :type sincal_interaction: boolean
    :param delete_files: Flag to delete already contained .sin files in the outputfolder
    :type delete_files: boolean
    :return: (Simulation Object (COM-Component), Sincal Document, Sincal Application Object)
    :rtype: Tuple
    '''
    if delete_files:
        # Dispatch in-process simulation object
        all_files = glob(os.path.join(output_folder, '*'))
        path = os.path.join(output_folder, file_name)
        if path in all_files:
            os.remove(path)
            shutil.rmtree(path.replace('.sin', '_files'))
            shutil.rmtree(output_folder)
        if not sincal_interaction:
            all_temp_files = glob(os.path.join(output_folder, '~*'))
            for path in all_temp_files:
                try:
                    os.remove(path)
                except PermissionError:
                    shutil.rmtree(path)
    try:
        simulation = win32com.client.Dispatch("Sincal.Simulation")
    except:
        raise RuntimeError("Error: Dispatch Simulation failed!")

    # Language
    simulation.Language("EN")

    if use_active_net is False:
        db_mgr = win32com.client.Dispatch("Sincal.DatabaseManager")

        filename, extenstion = os.path.splitext(file_name)
        db_path = os.path.join(output_folder, filename + '_files')
        dbfile = db_path + '\\database.db'
        project_name = os.path.join(output_folder, file_name)

        siDBTypeElectro = 1
        siDBParamLanguage = 1
        strNetwDB = r"TYP=NET;MODE=SQLite;FILE={};SINFILE={};".format(dbfile, project_name)

        db_mgr.SetConnection(strNetwDB)
        db_mgr.SetParameter(siDBParamLanguage, 'DE')

        iErr = db_mgr.CreateDB(siDBTypeElectro)
        if iErr != 0:
            print("Error: Database creation failed, error code:", iErr)

        db_info = "TYP=NET;MODE=SQLITE;FILE={};USR=Admin;PWD=;SINFILE={};".format(dbfile, project_name)

        ec = simulation.Database(db_info)

        ec = iErr | ec
        if ec == 0:
            print('Creation Sucessful')

        del db_mgr

        path = os.getcwd()
        if use_ui:
            # Obtain database connection string of the network model ("NET")
            sincal_app = win32com.client.Dispatch("SIASincal.Application")
            sincal_doc = sincal_app.OpenDocument(os.path.join(path, project_name))
        else:
            sincal_app = None
            sincal_doc = None
        time.sleep(0.5)
    else:
        # Obtain database connection string of the network model ("NET")
        sincal_app = win32com.client.Dispatch("SIASincal.Application")
        sincal_doc = sincal_app.GetActiveDocument()
        try:
            db_info = sincal_doc.Database("NET")
        except:
            db_info = "NET"
        #del sincal_app
        #sincal_app = None

        ec = simulation.Database(db_info)

        if ec == 0:
            print('Connection Sucessful')

    # Set network data to be loaded for the calculation (LF | SC)
    mask = 0x00000001 | 0x00000002#| 0x00200000 # time sereis calculation added
    simulation.SetInputState(mask)

    return simulation, sincal_doc, sincal_app


def initialize_net(simulation, output_folder, file_name):
    '''
    Intializes a sincal electrical Database Object.

    :param simulation: Simulation Object
    :type simulation: object
    :param output_folder: Path to the used folder
    :type output_folder: string
    :param file_name: Project name
    :type file_name: string
    :return: Sincal electrical Database Object
    :rtype: object
    '''
    # Set Batchmode to reference to on-disc database , e.g. 1
    simulation.BatchMode(1)
    simulation.LoadDB("LF")  # necessary to reference the database to the simulation object

    # Open electrical model (database) for modelling
    network_model = simulation.DB_EL()
    network_model.SetParameter("DBSYS_MODE", 8)  # 8 for SQlite
    ds = os.path.join(output_folder, file_name.replace('.sin', '_files'), 'database.db')
    network_model.SetParameter("DBSYS_DATAFILE", ds)
    network_model.OpenEx()
    return network_model


def initialize_calculation(net, simulation):
    '''
    Intializes a sincal electrical Database Object.

    :param net: Sincal electrical Database Object
    :type net: object
    :param simulation: Simulation Object
    :type simulation: object
    :return: None
    :rtype: None
    '''
    # Set Batchmode to reference to on-disc database , e.g. 1
    net.Close()
    simulation.BatchMode(0)
    simulation.LoadDB("LF")
    simulation.Start("LF")
    simulation.SaveDB("LF") # necessary to reference the database to the simulation object

    # Open electrical model (database) for modelling
    net.OpenEx()

