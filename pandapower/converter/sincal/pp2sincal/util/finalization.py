import os
import time

def write_to_net(net, doc):
    '''
    Closes the database object, reloads the sincal document (automation object) and deletes  doc in python.

    :param net: Sincal electrical Database Object
    :type net: object
    :param doc: Sincal Document (Automation Object)
    :type doc: object
    :return: None
    :rtype: None
    '''
    net.Close()
    if not doc is None:
        doc.Reload()
    del net
    time.sleep(0.5)


def close_net(app, sim, doc, output_folder, file_name):
    '''
    CLoses the sincal application object, deletes the items doc, app and sim in python.

    :param app: Sincal Application Object
    :type app: object
    :param sim: Simulation Object
    :type sim: object
    :param doc: Sincal Document (Automation Object)
    :type doc: object
    :param output_folder: Output folder for the converted network
    :type output_folder: string
    :param file_name: Name of the project. Must end with .sin.
    :type file_name: string
    :return: None
    :rtype: None
    '''
    if not app is None:
        app.CloseDocument(os.path.join(output_folder, file_name))
    del doc
    del app
    del sim
    time.sleep(0.5)
