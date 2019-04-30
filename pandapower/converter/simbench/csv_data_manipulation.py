import pandas as pd
from copy import deepcopy

try:
    import pplog as logging
except ImportError:
    import logging

from pandapower.converter.simbench.auxiliary import idx_in_2nd_array, avoid_duplicates_in_column

logger = logging.getLogger(__name__)

__author__ = 'smeinecke'


def _ensure_safe_csv_ids(csv_data):
    """ Ensures that no IDs are duplicated in csv_data. """
    if csv_data["Measurement"].shape[0] or csv_data["Switch"].shape[0]:
        not_allowed_tables_with_duplicates = pd.Series(["Node", "Coordinates", "Line",
                                                        "Transformer"])
    else:
        not_allowed_tables_with_duplicates = pd.Series(["Node", "Coordinates"])
    tables_with_duplicated_ids = [tablename for tablename in csv_data.keys() if "id" in csv_data[
        tablename].columns and sum(csv_data[tablename]["id"].duplicated())]
    if not_allowed_tables_with_duplicates.isin(tables_with_duplicated_ids).any():
        to_error = str(list(not_allowed_tables_with_duplicates.loc[
            not_allowed_tables_with_duplicates.isin(tables_with_duplicated_ids)]))
        raise ValueError("In " + to_error + " no duplicated IDs are allowed.")
    elif len(tables_with_duplicated_ids):
        for twdid in tables_with_duplicated_ids:
            avoid_duplicates_in_column(csv_data, twdid, "id")
        logger.info("In " + str(tables_with_duplicated_ids) + " duplicated IDs are renamed.")


def _ensure_single_switch_at_aux_node_and_copy_vm_setp(csv_data, new_type_name="node"):
    """
    This function set the Node type from 'auxiliary' to new_type_name for all nodes which are
    connected to multiple switches, because 'auxiliary' Nodes will be removed in
    create_branch_switches() while nodes with multiple switches are necessary for bus-bus switches.
    Furthermore, this function copies the vmSetp information from connected busbars to the nodes
    which got a new type name.
    """
    sw_nodes = pd.concat([csv_data["Switch"].nodeA, csv_data["Switch"].nodeB],
                         ignore_index=True)
    dupl_sw_node = sw_nodes[sw_nodes.duplicated()]
    dupl_node_ids = csv_data["Node"].id.isin(dupl_sw_node)
    aux_node_ids = csv_data["Node"].type == "auxiliary"
    idx_nodes_dupl_sw = csv_data["Node"].index[dupl_node_ids & aux_node_ids]

    # rename node type
    csv_data["Node"].type.loc[idx_nodes_dupl_sw] = new_type_name

    for X, Y in zip(["nodeA", "nodeB"], ["nodeB", "nodeA"]):
        # get indices to copy the setpoint
        node_names_dupl_sw = csv_data["Node"].id.loc[idx_nodes_dupl_sw]
        X_in_dupl = csv_data["Switch"][X].isin(node_names_dupl_sw)
        idx_Y = idx_in_2nd_array(csv_data["Switch"][Y].values,
                                    csv_data["Node"]["id"].values)
        Y_is_busbar = csv_data["Node"].loc[idx_Y].type.str.contains("busbar").values
        idx_in_sw_to_set = csv_data["Switch"].index[X_in_dupl & Y_is_busbar]

        idx_X = idx_in_2nd_array(csv_data["Switch"][X].loc[idx_in_sw_to_set].values,
                                    csv_data["Node"]["id"].values)
        idx_Y = idx_Y[X_in_dupl & Y_is_busbar]

        # only use the first setpoint for nodes which are connected to multiple busbars
        idx_X_pd = pd.Series(idx_X)
        idx_node_dupl = idx_X_pd.duplicated()

        # set setpoint
        csv_data["Node"].vmSetp.loc[idx_X[idx_node_dupl]] = csv_data["Node"].vmSetp.loc[
            idx_Y[idx_node_dupl]].values


# former used function without without copying vmSetp from busbar to retyped node:
#def _ensure_single_switch_at_aux_node(csv_data, new_type_name="node"):
#    """
#    This function set the Node type from 'auxiliary' to new_type_name for all nodes which are
#    connected to multiple switches, because 'auxiliary' Nodes will be removed in
#    create_branch_switches() while nodes with multiple switches are necessary for bus-bus switches.
#    """
#    sw_nodes = pd.concat([csv_data["Switch"].nodeA, csv_data["Switch"].nodeB],
#                         ignore_index=True)
#    dupl_sw_node = sw_nodes[sw_nodes.duplicated()]
#    dupl_node_ids = csv_data["Node"].id.isin(dupl_sw_node)
#    aux_node_ids = csv_data["Node"].type == "auxiliary"
#    csv_data["Node"].type.loc[dupl_node_ids & aux_node_ids] = new_type_name
    # naive coding:
#    aux_ids = csv_data["Node"].id[csv_data["Node"].type == "auxiliary"].values
#    multi_sw_ids = [aux_id for aux_id in aux_ids if sum(
#        (csv_data["Switch"].nodeA.values == aux_id) |
#        (csv_data["Switch"].nodeB.values == aux_id)) > 1]
#    multi_sw_idx = csv_data["Node"].index[csv_data["Node"].id.isin(multi_sw_ids)]
#    csv_data["Node"].type.loc[multi_sw_idx] = new_type_name


def _sort_switch_nodes_and_prepare_element_and_et(csv_data):
    """ 1) Swaps nodeA and nodeB data in switch table, if nodeA has auxiliary node names.
    As a result, no auxiliary node names are in nodeA column.
    2) Prepares "et" and "nodeB" (-> "element") columns for conversion to pp format. """
    # --- get indices/booleans
    idx_aux_node = csv_data["Node"].type == "auxiliary"
    aux_node_names = csv_data["Node"].id[idx_aux_node]
    nodeA_is_aux_node = csv_data["Switch"].nodeA.isin(aux_node_names)
    nodeB_is_aux_node = csv_data["Switch"].nodeB.isin(aux_node_names)
    both_are_aux_nodes = nodeA_is_aux_node & nodeB_is_aux_node
    if sum(both_are_aux_nodes):
        raise ValueError("In switch table, nodeA and nodeB are auxiliary node names in the" +
                         "indices: " + str(list(both_are_aux_nodes)))

    # --- swap nodeA data if there are auxiliary node names
    nodeA_data = deepcopy(csv_data["Switch"].nodeA[nodeA_is_aux_node])
    csv_data["Switch"].nodeA.loc[nodeA_is_aux_node] = csv_data["Switch"].nodeB[
        nodeA_is_aux_node]
    csv_data["Switch"].nodeB.loc[nodeA_is_aux_node] = nodeA_data

    # --- prepare 'element' and 'et' columns
    csv_data["Switch"]["et"] = "b"
    csv_data["Switch"]["nodeB"] = csv_data["Node"].index[idx_in_2nd_array(
        csv_data["Switch"]["nodeB"].values, csv_data["Node"]["id"].values)]


def _correct_autoTapSide_of_nonTapTrafos(csv_data):
    for elm in ["Transformer", "Transformer3W"]:
        nonTapTrafos = ~csv_data[elm].autoTap.astype(bool)
        csv_data[elm].autoTapSide.loc[nonTapTrafos] = None


def _add_phys_type_and_vm_va_setpoints_to_element_tables(csv_data):
    """ Creates 'phys_type', 'vm_pu' and 'va_degree' column in 'ExternalNet', 'PowerPlant', 'RES'
        tables as well as 'vm_from_pu' and 'vm_to_pu' in 'Line' table for dclines and autoTapSetp
        for trafos. """
    # --- "ExternalNet", "PowerPlant", "RES"
    for gen_table in ["ExternalNet", "PowerPlant", "RES"]:
        csv_data[gen_table]["phys_type"] = gen_table
        idx_node = idx_in_2nd_array(csv_data[gen_table].node.values, csv_data["Node"].id.values)
        csv_data[gen_table]["vm_pu"] = csv_data["Node"].vmSetp[idx_node].values
        csv_data[gen_table]["va_degree"] = csv_data["Node"].vaSetp[idx_node].values

    # --- Line (for dclines)
    for bus_type, param in zip(["nodeA", "nodeB"], ['vm_from_pu', 'vm_to_pu']):
        idx_node = idx_in_2nd_array(csv_data["Line"][bus_type].values,
                                       csv_data["Node"].id.values)
        csv_data["Line"][param] = csv_data["Node"].vmSetp[idx_node].values

    # --- Transformer, Transformer3W
    # will be done in _set_vm_setpoint_to_trafos() after create_branch_elements() to set the vm
    # setpoint from busbars and not from auxiliary nodes


def _extend_coordinates_to_node_shape(csv_data):
    """ Extends the Coordinates table to the shape of Nodes to enable copying simply to bus_geodata.
    """
    bus_geodata = pd.DataFrame([], index=csv_data["Node"].index, columns=["x", "y"])
    with_coord = ~csv_data["Node"]["coordID"].isnull()
    idx_in_coordID = idx_in_2nd_array(csv_data["Node"]["coordID"].loc[with_coord].values,
                                         csv_data["Coordinates"]["id"].values)
    bus_geodata.loc[with_coord, ["x", "y"]] = csv_data["Coordinates"].loc[idx_in_coordID,
                                                                          ["x", "y"]].values
    csv_data["Coordinates"] = bus_geodata
