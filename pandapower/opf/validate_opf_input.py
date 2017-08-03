
def _check_if_all_opf_parameters_are_given(net, logger):
    # Check if all necessary parameters are given:
    error = False
    controllable_elements = False
    elements_to_check = ["gen", "dcline", "sgen", "load"]
    for element in elements_to_check:
        if (not net[element].empty):
            if element == "gen":
                constraints_to_check = ["min_p_kw", "max_p_kw", "max_q_kvar", "min_q_kvar", "controllable"]
                controllable_elements = True
            elif element == "dcline":
                # ToDo: This seems untested!
                constraints_to_check = ["max_p_kw", "min_q_from_kvar", "max_q_from_kvar", "min_q_to_kvar", "max_q_to_kvar"]
                controllable_elements = True
            else:
                constraints_to_check = ["min_p_kw", "max_p_kw", "max_q_kvar", "min_q_kvar"]
                controllable_elements = net[element].controllable.any()
                logger.debug('No controllable %s found' % element)

            if controllable_elements:
                for constraint in constraints_to_check:
                    if constraint not in net[element].columns:
                        error = True
                        logger.error('Warning: Please specify %s constraints for controllable %s' % (constraint, element))
                    

    if error:
        raise KeyError("OPF parameters are not set correctly. See error log.")