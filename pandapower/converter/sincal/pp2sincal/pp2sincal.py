from pandapower.converter.sincal.pp2sincal.util.main import convert_pandapower_net, initialize, finalize

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

try:
    import simbench
except ImportError:
    logger.info(r'you need to install the package "simbench" first, if you want to run the simbench related scripts')

def pp2sincal(net_pp, output_folder, file_name, use_active_net=False, plotting=True, use_ui=False,
              sincal_interaction=False, delete_files=True, dc_as_sync=False, individual_fcts=None):
    '''
    Executes the three main steps, `initialize`, `convert_pandapower_net` and `finalize`.

    :param net_pp: The pandapower network that shall be converted
    :type net_pp: pandapowerNet
    :param output_folder: Output folder of the converted network
    :type output_folder: string
    :param file_name: Name of the project. Must end with .sin.
    :type file_name: string
    :param use_active_net: If you want to directly convert a pandapower net into an open Sincal instance set this \
            flag to True
    :type use_active_net: boolean
    :param plotting: Flag to graphically display elements
    :type plotting: boolean
    :param use_ui: If you want to open the Sincal user interface
    :type use_ui: boolean
    :param sincal_interaction: If you want to close the project in Sincal afterwards, set this flag to False
    :type sincal_interaction: boolean
    :param delete_files: Delete already contained .sin - files in the output folder
    :type delete_files: boolean
    :return: None
    :rtype: None
    '''
    net, app, sim, doc = initialize(output_folder, file_name, use_active_net, use_ui,
                                         sincal_interaction, delete_files)
    convert_pandapower_net(net, net_pp, doc, plotting, dc_as_sync)
    finalize(net, net_pp, output_folder, file_name, app, sim, doc, sincal_interaction, individual_fcts)


def convert_simbench_network(simbench_code, output_folder, use_active_net=False, plotting=True,
                             use_ui=False, sincal_interaction=False, delete_files=True):
    '''
    Converts a certain simbench network.

    :param simbench_code: Simbench Code for a certain network (for example '1-LV-urban6--2-sw'.
                                                               Get the codes from https://simbench.de/de/datensaetze/
    :type simbench_code: string
    :param output_folder: Output folder of the converted network
    :type output_folder: string
    :param use_active_net: If you want to directly convert a pandapower net into an open Sincal instance set this \
            flag to True
    :type use_active_net: boolean
    :param plotting: Flag to graphically display elements
    :type plotting: boolean
    :param use_ui: If you want to open the Sincal user interface
    :type use_ui: boolean
    :param sincal_interaction:  If you want to close the project in Sincal afterwards, set this flag to False
    :type sincal_interaction: boolean
    :param delete_files: Delete already contained .sin - files in the output folder
    :type delete_files: boolean
    :return: None
    :rtype: None
    '''
    net_pp = simbench.get_simbench_net(simbench_code)
    pp2sincal(net_pp, output_folder, simbench_code + '.sin', use_active_net, plotting, use_ui,
              sincal_interaction, delete_files)


def convert_all_simbench_networks(output_folder, plotting=True, use_ui=False,
                                  sincal_interaction=False, delete_files=True):
    '''
    Converts all simbench networks.

    :param output_folder: Output folder of the converted network
    :type output_folder: string
    :param plotting: Flag to graphically display elements
    :type plotting: boolean
    :param use_ui: If you want to open the Sincal user interface
    :type use_ui: boolean
    :param sincal_interaction: If you want to close the project in Sincal afterwards, set this flag to False
    :type sincal_interaction: boolean
    :param delete_files: Delete already contained .sin - files in the output folder
    :type delete_files: boolean
    :return: None
    :rtype: None
    '''

    simbench_codes = simbench.collect_all_simbench_codes()
    for simbench_code in simbench_codes:
        net_pp = simbench.get_simbench_net(simbench_code)
        pp2sincal(net_pp, output_folder, simbench_code + '.sin', False, plotting, use_ui,
                  sincal_interaction, delete_files)


def convert_simbench_networks_scenario(scenario, output_folder, plotting=True,
                                       use_ui=False, sincal_interaction=False, delete_files=True):
    '''
    Converts all simbench networks referring to a certain scenario.

    :param scenario: Choose one of three scenarios of each grid (0, 1, 2). The Scenarios represent the base state and
                     possible future states of the network ( 0: base, 1: 2024, 2: 2034)
    :type scenario: integer
    :param output_folder: Output folder of the converted network
    :type output_folder: string
    :param plotting: Flag to graphically display elements
    :type plotting: boolean
    :param use_ui: Use the open user interface
    :type use_ui: boolean
    :param sincal_interaction: If you want to close the project in Sincal afterwards, set this flag to False
    :type sincal_interaction: boolean
    :param delete_files: Delete already contained .sin - files in the output folder
    :type delete_files: boolean
    :return: None
    :rtype: None
    '''

    simbench_codes = simbench.collect_all_simbench_codes(scenario=scenario)
    for simbench_code in simbench_codes:
        net_pp = simbench.get_simbench_net(simbench_code)
        pp2sincal(net_pp, output_folder, simbench_code + '.sin', False, plotting, use_ui,
                  sincal_interaction, delete_files)


def convert_simbench_networks_level(level, output_folder, plotting=True,
                                    use_ui=False, sincal_interaction=False, delete_files=True):
    '''
    Converts simbench networks referring to a ceretain level.

    :param level: Voltage level for example 'EHV'
    :type level: string
    :param output_folder: Output folder of the converted network
    :type output_folder: string
    :param plotting: Flag to graphically display elements
    :type plotting: boolean
    :param use_ui: Use the open user interface
    :type use_ui: boolean
    :param sincal_interaction: If you want to close the project in Sincal afterwards, set this flag to False
    :type sincal_interaction: boolean
    :param delete_files: Delete already contained .sin - files in the output folder
    :type delete_files: boolean
    :return: None
    :rtype: None
    '''

    simbench_codes = simbench.collect_all_simbench_codes()
    for simbench_code in simbench_codes:
        if simbench_code.split('-')[1].lower() != level.lower():
            continue
        net_pp = simbench.get_simbench_net(simbench_code)
        pp2sincal(net_pp, output_folder, simbench_code + '.sin', False, plotting, use_ui,
                  sincal_interaction, delete_files)
