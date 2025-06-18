from pandapower.converter.powerfactory.pf_export_functions import run_short_circuit
from pandapower import pp_dir
import pandas as pd
import numpy as np
import os

testfiles_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests')


class PFShortCircuitAnalysis:
    def __init__(self, app, proj_name, fault_type='LLL', calc_mode='max',
                 fault_impedance_rf=0.0, fault_impedance_xf=0.0,
                 lv_tol_percent=10, fault_location_index=None, load_pp_net=False):
        """
                Parameters:
                - app: powerfactory.Application
                - proj_name: PowerFactory project name
                - fault_type: str ("LLL", "LL", "LG", "LLG")
                - calc_mode: str ("max", "min")
                - fault_impedance_rf: float
                - fault_impedance_xf: float
                - lv_tol_percent: float
                - fault_location_index: int or None
        """

        self.app = app
        self.proj_name = proj_name
        self.fault_type = fault_type
        self.calc_mode = calc_mode
        self.fault_impedance_rf = fault_impedance_rf
        self.fault_impedance_xf = fault_impedance_xf
        self.lv_tol_percent = lv_tol_percent
        self.fault_location_index = fault_location_index
        self.pf_results_bus_sc = None
        self.pf_results_branch_sc = None

        # activate project
        app.ActivateProject(proj_name)
        active_project = app.GetActiveProject()

        # activate study case
        study_cases = active_project.GetContents("Study Cases")
        berechnungsfaelle = active_project.GetContents("Berechnungsfälle")

        if study_cases:  # Check if the list is not empty
            study_case_folder = study_cases[0]
        else:
            # Set to None if both lists are empty
            study_case_folder = berechnungsfaelle[0] if berechnungsfaelle else None
        study_cases = study_case_folder.GetContents()
        study_case = study_cases[0]
        study_case.Activate()

    def get_pf_sc_bus_results(self):

        app = self.app
        fault_type = self.fault_type
        calc_mode = self.calc_mode
        fault_impedance_rf = self.fault_impedance_rf
        fault_impedance_xf = self.fault_impedance_xf
        lv_tol_percent = self.lv_tol_percent
        fault_location_index = self.fault_location_index

        res = run_short_circuit(app=app, fault_type=fault_type, calc_mode=calc_mode,
                                fault_impedance_rf=fault_impedance_rf, fault_impedance_xf=fault_impedance_xf,
                                lv_tol_percent=lv_tol_percent, fault_location_index=fault_location_index)
        if res == 1:
            raise UserWarning("short circuit results could not be calculated in powerfactory")

        bus_results = []
        bus_elements = app.GetCalcRelevantObjects('*.ElmTerm')

        result_variables_3ph = {
            "pf_ikss_ka": "m:Ikss",
            "pf_skss_mw": "m:Skss",
            "pf_rk_ohm": "m:R",
            "pf_xk_ohm": "m:X",  # ,"pf_ip_ka": "m:ip"
            "pf_vm_pu": "m:u1",
            "pf_va_degree": "m:phiui"
        }

        result_variables = {
            "pf_ikss_a_ka": "m:Ikss:A",
            "pf_ikss_b_ka": "m:Ikss:B",
            "pf_ikss_c_ka": "m:Ikss:C",
            "pf_skss_a_mw": "m:Skss:A",
            "pf_skss_b_mw": "m:Skss:B",
            "pf_skss_c_mw": "m:Skss:C",
            "pf_rk0_ohm": "m:R0",
            "pf_xk0_ohm": "m:X0",
            "pf_rk1_ohm": "m:R1",
            "pf_xk1_ohm": "m:X1",
            "pf_rk2_ohm": "m:R2",
            "pf_xk2_ohm": "m:X2",
            "pf_vm_a_pu": "m:ul:A",
            "pf_vm_b_pu": "m:ul:B",
            "pf_vm_c_pu": "m:ul:C",
            "pf_va_a_degree": "m:phiul:A",
            "pf_va_b_degree": "m:phiul:B",
            "pf_va_c_degree": "m:phiul:C"
        }

        if fault_type == 'LLL':
            result_variables = result_variables_3ph

        for bus in bus_elements:
            if bus.HasResults(0):
                bus_data = {'name': bus.loc_name}
                for col_name, pf_attribute in result_variables.items():
                    try:
                        value = bus.GetAttribute(pf_attribute)
                    except Exception:
                        value = np.nan
                    bus_data[col_name] = value
                bus_results.append(bus_data)

        self.pf_results_bus_sc = pd.DataFrame(bus_results)
        return self.pf_results_bus_sc

    def get_pf_sc_branch_results(self):

        app = self.app
        fault_type = self.fault_type
        calc_mode = self.calc_mode
        fault_impedance_rf = self.fault_impedance_rf
        fault_impedance_xf = self.fault_impedance_xf
        lv_tol_percent = self.lv_tol_percent
        fault_location_index = self.fault_location_index

        res = run_short_circuit(app=app, fault_type=fault_type, calc_mode=calc_mode,
                                fault_impedance_rf=fault_impedance_rf, fault_impedance_xf=fault_impedance_xf,
                                lv_tol_percent=lv_tol_percent, fault_location_index=fault_location_index)
        if res == 1:
            raise UserWarning("short circuit results could not be calculated in powerfactory")

        line_results = []
        line_elements = app.GetCalcRelevantObjects('*.ElmLne')

        result_variables_lines_3ph = {
            "pf_ikss_from_ka": "m:Ikss:bus1",
            "pf_ikss_to_ka": "m:Ikss:bus2",
            "pf_ip_from_ka": "m:Ip:bus1",
            "pf_ip_to_ka": "m:Ip:bus2",
            "pf_skss_from_mw": "m:Skss:bus1",
            "pf_skss_to_mw": "m:Skss:bus2",
            "pf_p_from_mw": "m:P:bus1",
            "pf_p_to_mw": "m:P:bus2",
            "pf_q_from_mvar": "m:Q:bus1",
            "pf_q_to_mvar": "m:Q:bus2",
            "pf_ikss_from_degree": "m:phii:bus1",
            "pf_ikss_to_degree": "m:phii:bus2"
        }

        result_variables_lines = {
            "pf_ikss_a_from_ka": "m:Ikss:bus1:A",
            "pf_ikss_b_from_ka": "m:Ikss:bus1:B",
            "pf_ikss_c_from_ka": "m:Ikss:bus1:C",
            "pf_ikss_a_to_ka": "m:Ikss:bus2:A",
            "pf_ikss_b_to_ka": "m:Ikss:bus2:B",
            "pf_ikss_c_to_ka": "m:Ikss:bus2:C",
            "pf_skss_a_from_mw": "m:Skss:bus1:A",
            "pf_skss_b_from_mw": "m:Skss:bus1:B",
            "pf_skss_c_from_mw": "m:Skss:bus1:C",
            "pf_skss_a_to_mw": "m:Skss:bus2:A",
            "pf_skss_b_to_mw": "m:Skss:bus2:B",
            "pf_skss_c_to_mw": "m:Skss:bus2:C",
            "pf_p_a_from_mw": "m:P:bus1:A",
            "pf_p_b_from_mw": "m:P:bus1:B",
            "pf_p_c_from_mw": "m:P:bus1:C",
            "pf_p_a_to_mw": "m:P:bus2:A",
            "pf_p_b_to_mw": "m:P:bus2:B",
            "pf_p_c_to_mw": "m:P:bus2:C",
            "pf_q_a_from_mvar": "m:Q:bus1:A",
            "pf_q_b_from_mvar": "m:Q:bus1:B",
            "pf_q_c_from_mvar": "m:Q:bus1:C",
            "pf_q_a_to_mvar": "m:Q:bus2:A",
            "pf_q_b_to_mvar": "m:Q:bus2:B",
            "pf_q_c_to_mvar": "m:Q:bus2:C",
            "pf_ikss_a_from_degree": "m:phii:bus1:A",
            "pf_ikss_b_from_degree": "m:phii:bus1:B",
            "pf_ikss_c_from_degree": "m:phii:bus1:C",
            "pf_ikss_a_to_degree": "m:phii:bus2:A",
            "pf_ikss_b_to_degree": "m:phii:bus2:B",
            "pf_ikss_c_to_degree": "m:phii:bus2:C"
        }

        if fault_type == 'LLL':
            result_variables_lines = result_variables_lines_3ph

        for line in line_elements:
            if line.HasResults(0):
                line_data = {'name': line.loc_name}
                for col_name, pf_attribute in result_variables_lines.items():
                    try:
                        value = line.GetAttribute(pf_attribute)
                    except Exception:
                        value = np.nan
                    line_data[col_name] = value
                from_bus = line.bus1.cterm
                to_bus = line.bus2.cterm

                if fault_type == 'LLL':
                    line_data["pf_vm_from_pu"] = from_bus.GetAttribute("m:u1")
                    line_data["pf_vm_to_pu"] = to_bus.GetAttribute("m:u1")
                    line_data["pf_va_from_degree"] = from_bus.GetAttribute("m:phiu")
                    line_data["pf_va_to_degree"] = to_bus.GetAttribute("m:phiu")
                else:
                    line_data["pf_vm_a_from_pu"] = from_bus.GetAttribute("m:u:A")
                    line_data["pf_vm_b_from_pu"] = from_bus.GetAttribute("m:u:B")
                    line_data["pf_vm_c_from_pu"] = from_bus.GetAttribute("m:u:C")
                    line_data["pf_vm_a_to_pu"] = to_bus.GetAttribute("m:u:A")
                    line_data["pf_vm_b_to_pu"] = to_bus.GetAttribute("m:u:B")
                    line_data["pf_vm_c_to_pu"] = to_bus.GetAttribute("m:u:C")
                    line_data["pf_va_a_from_degree"] = from_bus.GetAttribute("m:phiu:A")
                    line_data["pf_va_b_from_degree"] = from_bus.GetAttribute("m:phiu:B")
                    line_data["pf_va_c_from_degree"] = from_bus.GetAttribute("m:phiu:C")
                    line_data["pf_va_a_to_degree"] = to_bus.GetAttribute("m:phiu:A")
                    line_data["pf_va_b_to_degree"] = to_bus.GetAttribute("m:phiu:B")
                    line_data["pf_va_c_to_degree"] = to_bus.GetAttribute("m:phiu:C")

                line_results.append(line_data)

        self.pf_results_branch_sc = pd.DataFrame(line_results)
        return self.pf_results_branch_sc

    def get_case_name(self):
        fault_type = self.fault_type
        case = self.calc_mode
        fault_impedance = self.fault_impedance_rf, self.fault_impedance_xf
        lv_tol_percent = self.lv_tol_percent
        # fault_location = self.fault_location_index

        if fault_impedance[0] > 0:
            sheet_name_base = f"{fault_type.upper()}_{case}_fault"
        else:
            sheet_name_base = f"{fault_type.upper()}_{case}"
        sheet_name_base = f"{sheet_name_base}_{lv_tol_percent}"
        # if fault_location is not None:
        #     sheet_name_base = f"{sheet_name_base}_bus_{fault_location}"
        sheet_name_base = sheet_name_base[:25]
        return sheet_name_base
