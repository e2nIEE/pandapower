Change Log
=============

[upcoming release] - 2024-..-..
-------------------------------

[2.14.6] - 2024-04-02
-------------------------------
- [FIXED] more futurewarnings and deprecation warnings

[2.14.5] - 2024-03-28
-------------------------------
- [CHANGED] added possibility to provide custom weights to switches and transformers (before - always zero) when creating a graph
- [FIXED] many futurewarnings and deprecation warnings

[2.14.4] - 2024-03-28
-------------------------------
- [FIXED] internal pgm test returns ANOTHER error when using python 3.8
- [FIXED] added setuptools to relying tests

[2.14.3] - 2024-03-28
-------------------------------
- [FIXED] internal pgm test checked wrong result
- [FIXED] 2.14.0 - 2.14.3 just minor release fixes to improve workflow

[2.14.0] - 2024-03-26
-------------------------------
- [ADDED] function to search std_types from the basic standard type library
- [ADDED] Documentation for running powerflow using power-grid-model
- [ADDED] exporting to :code:`GeoJSON` with all properties from :code:`bus`, :code:`res_bus` and :code:`line`, :code:`res_line`
- [ADDED] function to run powerflow using the power-grid-model library
- [FIXED] short-circuit calculation: wrong results when :code:`ext_grid` and :code:`gen` elements are connected to the same bus
- [ADDED] comparison of "dict" attributes in :code:`pandapower.toolbox.comparisons.nets_equal` with :code:`DeepDiff`
- [FIXED] loading net from xlsx with MultiIndex
- [FIXED] numba version check
- [FIXED] deprecation warnings for numba (set nopython=True in jit calls)
- [FIXED] setting MultiIndex when loading empty DataFrame from JSON, getting next index from DataFrame with MultiIndex
- [FIXED] some fixes and small updates at cim2pp
- [CHANGED] add numba in the dependencies for Python 3.11 for GitHub test and release actions; revise numba version checks
- [ADDED] improved documentation for short-circuit calculation (description of the function and the element results)
- [FIXED] bug in :code:`pp.select_subnet` when using tap dependent impedance
- [ADDED] extensive unit tests for cim2pp converter (element parameter and load flow results)
- [FIXED] bug in :code:`cim2pp.build_pp_net` when setting default values for converted xwards
- [FIXED] bug in :code:`cim2pp.build_pp_net` when controller for gen is at TopologicalNode instead of ConnectivityNode
- [CHANGED] adjust default iterations for runpp_3ph
- [CHANGED] always convert RATE_A to ppc in build_branch (not only when mode == 'opf' as before)
- [FIXED] in converter from PowerFactory, collect all buses (even not relevant for the calculation) for connectivity issues
- [FIXED] bug in coords conversion in cim2pp, small fixes
- [CHANGED] cim2pp: added support for multi diagram usage for DL profiles
- [CHANGED] cim2pp: made build_pp_net modular by introducing classes
- [ADDED] cim2pp: added option to opt out of internal powerflow calculation
- [FIXED] error handling in :code:`plotly/mapbox_plot.py` not raising :code`ImportError` if :code:`geopy`  or :code:`pyproj` are missing
- [FIXED] powerfactory2pandapower-converter error if a line has two identical coordinates
- [ADDED] logger messages about the probabilistic load flow calculation (simultaneities) in the powerfactory2pandapower-converter for low voltage loads
- [ADDED] matplotlib v3.8.0 support (fixed :code:`plotting_colormaps.ipynb`)
- [CHANGED] PowerFactory converter - name :code:`for_name` as :code:`equipment` for all elements; also add to line
- [ADDED] option to use a second tap changer for the trafo element
- [CHANGED] parameters of function merge_internal_net_and_equivalent_external_net()
- [FIXED] :code:`convert_format.py`: update the attributes of the characteristic objects to match the new characteristic
- [FIXED] fixed the wrong id numbers for pypower powerflow algorithms fdxb and fdbx
- [FIXED] additional arguments from mpc saved to net._options: create "_options" if it does not exist
- [CHANGED] cim2pp: extracted getting default classes, added generic setting datatypes from CGMES XMI schema
- [ADDED] function :code:`getOTDF` to obtain Outage Transfer Distribution Factors, that can be used to analyse outages using the DC approximation of the power system
- [ADDED] function :code:`outage_results_OTDF` to obtain the matrix of results for all outage scenarios, with rows as outage scenarios and columns as branch power flows in that scenario
- [FIXED] add some safeguards for TDPF to avoid numerical issues in some cases
- [CHANGED] numba version check during init phase, not during calculation, saving about 3% calculation time for a loadflow
- [FIXED] avoid attaching elements as duplicates to a group where some of the elements already exist
- [ADDED] the function :code:`run_contingency` can raise a captured error if parameter :code:`raise_errors` is passed
- [FIXED] bugfix for tap dependent impedance characteristics so that not all characteristics columns are necessary
- [ADDED] add kwargs passing of get_equivalent() to runpp_fct()
- [ADDED] auxiliary functions ets_to_element_types() and element_types_to_ets() as well as toolbox function get_connected_buses_at_switches() and extension to get_connected_switches()
- [FIXED] in function :code:`toolbox.replace_zero_branches_with_switches`, use absolute for the parameters of impedance elements in case they are negative nonzero values
- [FIXED] in :code:`reindex_elements`: fixed index error when reindexing line_geodata
- [FIXED] bug in :code:`cim2pp`: Changed zero prioritized generators with voltage controller to sgens (like PowerFactory does)
- [ADDED] cim2pp: added description fields for each asset and added BusbarSection information to nodes
- [CHANGED] cim2pp: reformat documentation for reading in files
- [CHANGED] allow providing grid_tables as a parameter to the function that downloads net from PostgreSQL
- [FIXED] avoid FutureWarning of pandas 2.2
- [FIXED] compatibility with lightsim2grid after new version 0.8.0
- [ADDED] allow passing custom runpp-function to pp.diagnostic

[2.13.1] - 2023-05-12
-------------------------------
- [FIXED] missing test files for CIM converter test in the release files


[2.13.0] - 2023-05-12
-------------------------------
- [FIXED] another correction of shunt values in CIGRE HV
- [FIXED] deprecated np.typedict to np.sctypedict in cim converter
- [ADDED] reporting for cim2pp converter
- [ADDED] interfaces for repair functions for cim2pp converter
- [ADDED] using PandaModels to optimize reactive power provision for loading reduction
- [FIXED] several bugs in cim2pp converter, e.g. non linear tap changer issue
- [FIXED] shape issues when calculating SC with the superposition method
- [FIXED] typos in cim2pp tutorial
- [FIXED] creating geo coordinates form GL profile when ConnectivityNode is only in tp/tp_bd profile for cim2pp converter
- [FIXED] bugfix in _get_bus_v_results where vm_pu was not set for DC power flow, leading to old results staying in the bus results table
- [ADDED] simple cim2pp converter test
- [CHANGED] run ac pf instead of dc pf in estimation when parameter fuse_buses_with_bb_switch != 'all'
- [REMOVED] support for deprecated functions in :code:`groups.py`: :code:`check_unique_group_names`, :code:`append_to_group`


[2.12.1] - 2023-04-18
-------------------------------
- [FIXED] add minimum Python version (3.8) explicitly to setup.py
- [FIXED] remove :code:`import pandapower.test` from :code:`__init__`
- [FIXED] matplotlib imports are optional (but required for plotting)
- [FIXED] missing numpy int imports
- [FIXED] documentation; needed change: group functions parameter :code:`raise_` is renamed by :code:`raise_error`

[2.12.0] - 2023-04-06
-------------------------------
- [ADDED] feature: storing to json and restoring of nets with pandas multiindex dataframes and series
- [ADDED] several 'marker size legends' sizes + a spec. color can be passed to weighted_marker_traces
- [CHANGED] changed default optimization method in the estimation module from OptAlgorithm to "Newton-CG"
- [CHANGED] cim2pp converter documentation fixes
- [CHANGED] make legend item size constant in :code:`simple_plotly`
- [FIXED] add (nan) field "coords" to bus geodata in create_cigre_network_hv to avoid fatal error when exporting to Excel
- [FIXED] documentation of powerfactory converter
- [FIXED] create.py: if optional arguments are None or nan, the optional columns will not be added
- [FIXED] add tap_dependent_impedance attributes to trafo3w instead of trafo, in create.create_transformer3w and create.create_transformer3w_from_parameters
- [CHANGED] renamed functions: drop_from_group() -> detach_from_group(), append_to_group() -> attach_to_group(), check_unique_group_names() -> check_unique_group_rows()
- [CHANGED] attach_to_group(): enable handling of different reference_column passed than existing
- [ADDED] toolbox function :code:`count_elements`, :code:`drop_elements`, :code:`res_power_columns`
- [ADDED] new group functions :code:`element_associated_groups`, :code:`attach_to_groups`, :code:`group_res_power_per_bus`, :code:`group_index`
- [CHANGED] __repr__ (used by print(net)) now considers groups appropriately
- [ADDED] documentation of DeprecationWarning process
- [ADDED] add TDPF parameters as optional parameters for create line functions in create.py
- [CHANGED] remove support for Python 3.7 and add Python 3.11
- [CHANGED] split toolbox.py -> better overview, avoiding circular imports
- [CHANGED] aim for toolbox parameter name consistency: element_types, element_index (changes to mandatory parameters only)
- [CHANGED] output type of toolbox function :code:`element_bus_tuples`: set -> list
- [ADDED] import of internal packages such as control or converter
- [ADDED] group consideration in toolbox replace element functionality
- [ADDED] implementation of the "recycle" functionality for DC power flow and timeseries with run=pp.rundcpp
- [ADDED] calculate branch results for current magnitude and angle, voltage magnitude and angle, active and reactive power for short circuit calculation
- [ADDED] implement the superposition method ("Type C") for the short circuit calculation - consider pre-fault voltages
- [FIXED] Trafo control stepping direction for side=="hv"
- [ADDED] feature: protection - implementation of over-current relay
- [FIXED] Shunt admittance modelling for 3 phase calculations
- [ADDED] bulk creation function create_storages and create_wards
- [ADDED] FACTS devices Shunt Var Compensator (SVC) and Thyristor-Controlled Series Capacitor (TCSC) as new pandapower elements net.svc and net.tcsc

[2.11.1] - 2023-01-02
-------------------------------
- [ADDED] a 'marker size legend' (scale_trace) can be displayed for weighted_marker_traces with plotly
- [FIXED] bugfix in toolbox._merge_nets_deprecated
- [CHANGED] added tests for pp.control.Characteristic, removed Characteristic.target

[2.11.0] - 2022-12-14
-------------------------------
- [ADDED] plotting function for dclines (create_dcline_collection), also added in simple_plot
- [ADDED] calculation of overhead line temperature in Newton-Raphson with two simplified methods (Frank et al. and Ngoko et al.)
- [ADDED] group functionality
- [ADDED] auxiliary function warn_and_fix_parameter_renaming to throw a derpecation warning (not an Error) if old name of a parameter is used
- [ADDED] zero-sequence parameters for net.impedance
- [ADDED] File I/O: Can now save and load pandapower serializable objects to Excel, PostgreSQL
- [ADDED] additional_traces (prepared by the user) can be passed to simple_plotly
- [ADDED] Added converter CGMES v2.4.15 to pandapower
- [CHANGED] TDPF: rename r_theta to r_theta_kelvin_per_mw, add r_theta_kelvin_per_mw to net.res_line
- [CHANGED] Compatibility with pandas 1.5, dropped "six" dependency
- [CHANGED] from_ppc(): revision of indexing and naming of elements
- [CHANGED] Complete revision of validate_from_ppc()
- [ADDED] helper functions for contingency calculation
- [CHANGED] Improve defaults, add docstrings and rename parameters of plot_voltage_profile() and plot_loading()
- [CHANGED] merge_nets(): revised for groups and new behavior regarding indexing; reindex_elements(): revised for groups, don't overwrite column "index" and feature parameter lookup
- [FIXED] Bug with user_pf_options: _init_runpp_options in auxiliary.py ignored user_pf_options when performing sanity checks

[2.10.1] - 2022-07-31
-------------------------------
- [FIXED] remove the parameter ignore_order in DeepDiff (__eq__), add __hash__ to JSONSerializableClass
- [ADDED] store and restore functionality of dataframe index names with to_json() and from_json()
- [ADDED] generalization from_json() with parameter empty_dict_like_object

[2.10.0] - 2022-07-29
-------------------------------
- [ADDED] added arbitrary keyword arguments, ``**kwargs``, in all create-functions
- [ADDED] groups functionality to allow grouping pandapower net elements and enable functionality to such groups
- [FIX] from_ppc() converter and power system test cases: add missing factor for tap_side=="lv"; change tap_side to "hv" for all test cases (were converted without new factor, so as the tap_side is "hv")
- [ADDED] from_mpc() converter: added functionality to import .m files via external package
- [CHANGED] from_ppc() converter: added option of tap_side and essential speed up
- [CHANGED] drop support of pandas versions < 1.0
- [ADDED] parameter in_ka for rated switch current
- [ADDED] current and loading result for switches
- [FIXED] bug for disabled continous tap controllers
- [ADDED] File I/O download and upload pandapowerNet to PostgreSQL
- [ADDED] __eq__ method for JSONSerializableClass using deepdiff library, and adjusted pp.nets_equal to use it. Requires deepdiff
- [CHANGED] enable calculating PTDF for a subset of branches
- [ADDED] in from_json one can pass a new variable of type dict called 'replace_elements'. Dict values replace the key in the loaded json string.

[2.9.0]- 2022-03-23
----------------------
- [ADDED] added support for Python 3.10
- [ADDED] added a function to pandapower.plotting to set line geodata from the geodata of the connected buses
- [ADDED] plotly hover information will indicate parallel lines, if parallel > 1
- [ADDED] 'showlegend' option for simple_plotly
- [CHANGED] rename u by vm (voltage magnitude) in file and functions names
- [FIX] plotly: only one legend entry for all lines/trafos instead of single entries for each one
- [FIX] fixed deprecation warning for pandas indexing with set (sets changed to lists inside .loc)
- [FIX] fixed hover information for lines in plotly
- [ADDED] functions to obtain grid equivalents (power system reduction with REI, Ward, X-Ward methods)
- [CHANGED] use numpy to vectorize trafo_control
- [ADDED] generic functions in pandapower.toolbox to read and write data to/from elements
- [CHANGED] remove code duplication in const_control, characteristic_control
- [ADDED] added the functionality to import grid data from PowerFactory
- [FIXED] failing tests for PowerModels integration due to the missing pm option "ac"

[2.8.0]- 2022-02-06
----------------------
- [ADDED] toolbox functions false_elm_links() and false_elm_links_loop()
- [FIXED] poly_cost and pwl_cost consideration in merge_nets()
- [ADDED] "results" initialization for runopp()
- [CHANGED] toolbox function nets_equal()
- [ADDED] toolbox function merge_same_bus_generation_plants()
- [ADDED] new object table "characteristic", new class "Characteristic" and "SplineCharacteristic" that are callable and return a value based on input according to a specified curve
- [FIXED] toolbox replace_ward_by_internal_elements() index usage
- [ADDED] TapDependentImpedance controller that adjusts the transformer parameters (e.g. vk_percent, vkr_percent) according to the tap position, based on a specified characteristic
- [ADDED] tap dependent impedance internally in build_branch: transformer (2W, 3W) parameters (e.g. vk_percent, vkr_percent) are adjusted according to the tap position based on a specified characteristic in the optional columns
- [ADDED] multiple costs check in create functions and runopp
- [ADDED] correct_dtypes() function for fileIO convert
- [FIXED] revise to_ppc() and to_mpc() init behaviour
- [CHANGED] import requirements / dependencies
- [ADDED] with the option "distributed_slack" for pp.runpp: distributed slack calculation to newton-raphson load flow; new column "slack_weights" for ext_grid, gen and xward; only 1 reference bus is allowed, any further reference buses are converted to PV buses internally
- [CHANGED] improved the integration with the package lightim2grid (fast power flow backend written in C++), add the test coverage for using lightsim2grid (for both versions, single slack and distributed slack, see https://lightsim2grid.readthedocs.io/en/latest/ on how to install and use lightsim2grid) #1455
- [FIXED] checks for when to activate and deactivate lightsim2grid in pp.runpp, added tests
- [ADDED] from_mpc: import additional variables from MATPOWER file as keys in net._options
- [FIXED] output_writer: bugfix for "res_{element}_3ph" to also run timeseries with runpp_3ph
- [FIXED] DeprecationWarning in pandas: use pandas.Index instead of pandas.Int64Index
- [FIXED] scipy version requirement: cancel the version limit
- [CHANGED] drop support for Python 3.6
- [FIXED] bugfix in timeseries calculations with recycle=True #1433
- [CHANGED] run tests in GuitHub Actions for pull requests to all branches
- [FIXED] net.unser_pf_options: bugfix for overruling the parameters that are in user_pf_options
- [ADDED] add_zero_impedance_parameters(): convenience function to add all required zero-sequence data for runpp_3ph from std_types and apply realistic assumptions
- [CHANGED] adjusted create.py functions to also include zero-sequence parameters
- [CHANGED] new tutorials for the voltage deviation model and the power flow calculation with PowerModels.jl
- [CHANGED] create_lines: enable batch creating of multiple lines now with multiole std_type entries instead of using the same std_type
- [CHANGED] OPF parameter "OPF_FLOW_LIM" now accessible through kwargs
- [CHANGED] Included DC line elements and results in to_html
- [FIXED] bugfix for currents of transformers in 3ph power flow #1343
- [CHANGED] check the dtype of the tap_pos column in the control_step of the transformer controller #1335
- [FIXED] net.sn_mva corrected for power_system_test_cases #1317
- [FIXED] fixed bugs in automatically identifying power station units (short-circuit calculation enhancements are still in progress)

[2.7.0]- 2021-07-15
----------------------
- [ADDED] Optimized the calculation of single/selected buses in 1ph/2ph/3ph short-circuit calculation
- [ADDED] Power station units with gen and trafo designated with "ps_trafo_ix" for short-circuit calculation
- [ADDED] Multiple example networks and network variations from IEC 60909-4
- [ADDED] OR-Tools implementation of linprog solver
- [ADDED] Efficient PTDF calculation on large grid
- [ADDED] toolbox function replace_pq_elmtype()
- [ADDED] Alternative constructor for DiscreteTapControl to use net.trafo.tap_step_percent to determine vm_lower_pu and vm_upper_pu based on vm_set_pu
- [ADDED] Characteristic object that represents a piecewise-linear characteristic
- [ADDED] CharacteristicControl that implements adjusting values in net based on some other input values in the grid
- [ADDED] USetTapControl that adjusts the setpoint for a transformer tap changer, based on a specified result variable (e.g. i_lv_ka)
- [CHANGED] Short-circuit gen calculation parameter "rkss_pu" to "rkss_ohm" according to IEC 60909 example
- [CHANGED] ConstControl can now also change attributes of other controllers, if the parameter "variable" is defined in the format "object.attribute" (e.g. "object.vm_set_pu")
- [CHANGED] ConstControl is initialized with level=-1 and order=-1 by default to make sure that it runs before other controllers
- [CHANGED] ConstControl now writes values from the datasource to net at time_step instead of control_step, which ensures that the values for the time step are set before running the initial power flow
- [CHANGED] replaced naming for "inductive" or "ind" by "underexcited" and "capacitive" or "cap" for "overexcited"

[2.6.0]- 2021-03-09
----------------------
- [ADDED] Factorization mode instead of inversion of Ybus in short-circuit calculation.
- [ADDED] Optimized the calculation of single/selected buses in 1ph/2ph/3ph short-circuit calculation.
- [ADDED] New options for run_control to 'continue on divergence' and 'check each level' PR #1104.
- [ADDED] Check for necessary and valid parameters to calculate 3ph powerflow.
- [ADDED] Toolbox method get_connecting_branches to determine branches which connect two sets of buses.
- [CHANGED] Deleting set_q_from_cosphi from ConstControl and deprecation warning. Use a separate ConstControl for setting Q timeseries instead.
- [CHANGED] Removed official Python 3.5 support due to end of its life #994.
- [FIXED] matching_params was missing in basic controller.
- [FIXED] Order of latitude and longitude in plotly mapbox plot.
- [FIXED] Dependencies of powerflow result plotting.
- [FIXED] init_ne_line to work with switches and parallel lines. Needed for PowerModels TNEP.

[2.5.0]- 2021-01-08
----------------------
- [ADDED] github actions for tests added.
- [ADDED] tests for PowerModels.jl interface (julia tests).
- [ADDED] documentation on how to install Gurobi as a PowerModels.jl solver.
- [ADDED] the voltage set point of external grids can now be optimized by the OPF by setting net.ext_grid.controllable to True.
- [ADDED] the Powermodels AC OPF can now be used with line loading constraints formulated with respect to the maximum current net.line.max_i_ka by using  pp.runpm_ac_opf(net, opf_flow_lim="I").
- [ADDED] for easier debugging of the Powermodels interface, you can now save your .json file and specify the file name by using pp.runpm(net, delete_buffer_file=False, pm_file_path="filename.json").
- [CHANGED] The create-module now contains some functions for standardized checks and procedures in all create functions.
- [CHANGED] all controllers and output writers do not have net as attribute any more.
- [CHANGED] due to multi net implementations in pandapipes, time series functions have been adapted drastically in order to minimize duplicated code.
- [CHANGED] internal data structure tutorial contains now an example of a spy plot to visualize the admittance matrix Ybus.
- [CHANGED] introduce abstract node/branch formulation for the plotly functions.
- [FIXED] issue # 905 fixed (If powerflow not necessary, e.g. two ext_grids/pv-nodes with only two buses) powerflow is bypassed and the solution is trivial.
- [FIXES] issue # 954 fixed (Update bus IDs for net.asymmetric_load and net.asymmetric_sgen when merging nets in toolbox.py).
- [FIXED] issue # 780 fixed (passing the shape to pypower solves the problem)
- [FIXED] excel engine pd.ExcelFile not working in new pandas version. Adaptation in file_io with new module openpyxl. openpyxl needs to be installed. Requirements are adapted accordngly.
- [FIXED] in io_utils functions with no clear class name can be de-serialized as well.
- [FIXED] fixed generic coordinates creation when respect_switches is set.
- [FIXED] recycle values None and False are considered equally --> recycle usage is skipped.
- [FIXED] control_diagnostic distinguishes between two winding and three winding transformers.
- [FIXED] toolbox functions, e.g. get_connected_elements, consider switches for three winding transformers.
- [FIXED] json load for broken geom columns in bus_geodata.

[2.4.0]- 2020-09-01
----------------------
- [CHANGED] signing system in state estimation: bus p,q measurement in consumption reference (load is positive) #893
- [ADDED] new element "net.motor" to model asynchronous machines #244
- [ADDED] possibility to calculate all branch currents in short-circuit calculations #862
- [ADDED] more flexibility in the create_generic_geodata function

[2.3.1]- 2020-08-19
----------------------
- [ADDED] Missing dependencies xlswriter, xlrd, cryptography
- [FIXED] Bug in rundcpp result table initialization
- [CHANGED] PTDF/LODF calculation to improve performance
- [FIXED] Signing system for P/Q values in net.res_bus_3ph
- [FIXED] JSON I/O handling of controllers with NaN values

[2.3.0]- 2020-08-11
----------------------
- [ADDED] Create functions for multiple gens, sgens, lines, trafos and switches
- [ADDED] Unbalanced power flow runpp_3ph
- [ADDED] Zero sequence power flow models for ext_grid, transformer, line, asymmetric_load, asymmetric_sgen
- [ADDED] Minimal 1ph fault calculation according to IEC 60909
- [CHANGED] OPF calculate_voltage_angles defaults to True instead of False
- [ADDED] lightsim2grid interface in NR power flow thanks to @BDonnot https://github.com/BDonnot/lightsim2grid
- [FIXED] PowerModels.jl solver interface call functions. Added OPFNotConverged to Powermodels.jl call
- [FIXED] pandas 1.0 and 1.1 support
- [CHANGED] revision of toolbox function drop_out_of_service_elements()
- [ADDED] toolbox function drop_measurements_at_elements()
- [ADDED] Encryption for JSON I/O
- [FIXED] Bug in converting measurements of out-of-service branch in state estimation #859
- [FIXED] Bug in using initialization option "results" in state estimation #859
- [CHANGED] In state estimation power flow results will not be renamed anymore
- [ADDED] New feature for defining the number of logging columns for an eval_function of an outputwriter log variable. Example: See log_variable docstring

[2.2.2]- 2020-03-17
----------------------
- [CHANGED] reset_results empties result tables per default
- [CHANGED] nan values result tables of power system test cases are emptied
- [ADDED] dclines and considering given branch indices by create_nxgraph()
- [ADDED] use_umfpack and permc_spec option from scipy spsolve in Newton-Raphson power flow
- [FIXED] Changed the __deepcopy__ for pandapowerNet back to using copy.deepcopy, fixed the issue that caused the switch to json #676
- [FIXED] Potential memory leaks due to circular references in JSONSerializableObjects, fixed by using weakref #677

[2.2.1]- 2020-01-29
----------------------
- [FIXED] Missing csv files #625
- [FIXED] deepcopy speed and missing DataFrames in net #620, #631
- [FIXED] simple plotly error with generic coords #619
- [FIXED] create line with passed geodata #610
- [FIXED] ConstControl write to and all_index attribute #609
- [FIXED] collection plotting issue #608


[2.2.0]- 2020-01-17
----------------------
- [ADDED] control and timeseries module
- [ADDED] Support phasor measurement in state estimation
- [ADDED] Support recycle in state estimation
- [ADDED] PowerModels.jl converter callable without running the PowerModels optimization
- [ADDED] Other PowerModels features via interface callable (e.g. network data check and different solver)
- [ADDED] toolbox function select_subnet now also copies cost data and net parameters
- [ADDED] toolbox functions replace_ward_by_internal_elements and replace_xward_by_internal_elements
- [ADDED] consideration of result tables in toolbox functions drop
- [ADDED] new jupyter notebook examples for time series, controller and PowerModels.jl interface
- [ADDED] reindex_buses() toolbox function

- [FIXED] Bugfixes in PowerModels conversion, OPF in general and tests
- [FIXED] renew opf_task() toolbox function which got outdated
- [FIXED] dtype at element parameter in cost tables
- [FIXED] convert_format.py: added the renaming of controller column and of the controller attributes, added tests for version 2.1.0

- [CHANGED] Unified the mesurement unit conversion of state estimation in ppc conversion
- [CHANGED] OPF bounds and settings for gens. limits or fixed values can now be enforced. See #511
- [CHANGED] OPF documentation and _check_necessary_opf_parameters()
- [CHANGED] JSON I/O: pandapower objects that are derived from JSONSerializableClass are now instantiated using __new__ instead of __init__ (as before), and the serialization has been adjusted; self.update_initialized(locals()) is not necessary anymore and has been removed; restore_json_objects is not needed anymore and has been removed
- [CHANGED] column name in net.controller: "controller" -> "object"
- [CHANGED] variable names in ContinuousTapControl ("u_set" -> "vm_set_pu") and in DiscreteTapControl ("u_lower" -> "vm_lower_pu", "u_upper" -> "vm_upper_pu")
- [CHANGED] __version__ is now changed to 2.2.0

[2.1.0]- 2019-07-08
----------------------
- [ADDED] calc_single_sc function to analyse a single fault instead of vectorized fault
- [ADDED] convenience function for logarithmic colormaps in plotting
- [CHANGED] corrected spelling 'continous' to 'continuous' in several functions
- [ADDED] additional standard types for overhead lines
- [CHANGED] make pp.to_json format closer to the JSON standard #406
- [ADDED] PowerModels.jl storage interface for time series based storage optimization.
- [ADDED] PowerModels.jl OTS interface for optimize transmission switching optimization.
- [ADDED] PowerModels.jl TNEP interface for transmission expansion optimization. See Jupyter Notebook
- [ADDED] pytest slow marker for tests and functions to run all, slow or fast tests
- [ADDED] Graph-Tool interface
- [ADDED] Multiple new algorithms and robust estimators in state estimation
- [ADDED] Support measurements for trafo3w in state estimation
- [ADDED] Auto zero-injection bus handling in state estimation

[2.0.1]- 2019-03-28
----------------------
- [FIXED] bug in short-circuit impedance of gens
- [ADDED] use estimation of rdss_pu defined in IEC 60909 of gens if not defined

[2.0.0]- 2019-03-21
----------------------
- [CHANGED] units from kW/kVAr/kVA to MW/MVAr/MVA in all elements #73
- [CHANGED] signing system from load to generation in gen, sgen and ext_grid #208
- [CHANGED] all trafo tap parameters from 'tp' to 'tap', tp_mid to tap_neutral #246
- [CHANGED] all trafo short-circuit voltage parameter names from "vsc" to "vk" #246
- [CHANGED] definition of cost functions #211
- [CHANGED] definition of measurements in measurement table #343
- [ADDED] interface to PowerModels.jl for OPF #207
- [CHANGED] removed Python 2 support #224
- [ADDED] load flow and OPF for user-defined temperature of lines, with the optional columns in line table "alpha" and "temperature_degree_celsius" #283
- [ADDED] z_ohm parameter in net.switch to assign resistance to switches #259
- [FIXED] initializing from results also considers auxiliary buses #236
- [ADDED] trafo3w switches are supported in create_nxgraph #271
- [CHANGED] create_nxgraph adds edges in multigraph with key=(element, idx) instead of key=0,1.. #85
- [CHANGED] patch size in create_bus_collection is not duplicated for rectangles anymore #181

[1.6.1] - 2019-02-18
----------------------
- [CHANGED] Patch size in create_bus_collection is not duplicated for rectangles anymore #181
- [CHANGED] Mask colormap z array to ensure nan handling
- [FIXED] active power distribution in DC OPF for multiple generators at one bus
- [ADDED] support for networkx graphs in json IO
- [ADDED] support for shapely objects in json IO
- [ADDED] switches for three winding transformers #30
- [ADDED] net.bus_geodata.coords to store line representation of busbars and create_busbar_collection to plot them
- [CHANGED] draw_collections also supports tuples of collections
- [ADDED] OPF logging output for verbose=True
- [ADDED] compatibility for pandas 0.24
- [FIXED] bug for single bus networks in DC PF #288

[1.6.0] - 2018-09-18
----------------------
- [CHANGED] Cost definition changed for optimal powerflow, see OPF documentation (http://pandapower.readthedocs.io/en/v1.6.0/powerflow/opf.html) and opf_changes-may18.ipynb
- [ADDED] OPF data (controllable, max_loading, costs, min_p_kw, ...) in Power System Test Cases
- [ADDED] case_ieee30, case5, case_illinois200
- [FIXED] 1 additional Trafo in case39, vn_kv change in case118, sgen indices in polynomial_cost in case 1888rte, case2848rte
- [ADDED] toolbox functions replace_impedance_by_line(), replace_line_by_impedance() and get_element_indices() including tests
- [CHANGED] new implementation of to_json, from_json for loading and saving grids using functools.singledispatch
- [FIXED] checking similar to "if x: ..." or "x = x or ..." when it is meant "if x is None: ...", because it is potentially problematic with some types
- [FIXED] convert_format: some older pandapower grids had "0" as "tp_side" in net.trafo, this is checked now as well
- [FIXED] create_buses: accepts a single tuple (set the same geodata for all buses) or an array of the corresponding shape (for individual geodata)
- [CHANGED] create_ext_grid_collection (plotting): ext_grid and ext_grid buses can be specified if a collection should only include some of ext grids
- [ADDED] ability to define phase shifting transformers with tp_st_percent #117
- [ADDED] support for multiple voltage controlling elements (ext_grid, gen, dcline) at one bus #134
- [CHANGED] reduced number of arguments in runpp by moving some less important arguments to kwargs #122
- [ADDED] parameters init_vm_pu and init_va_degree to allow independent initialization of bus magnitude and angle #113
- [ADDED] number of power flow iterations are now saved
- [ADDED] calculation of r, x and z for networkx branches
- [ADDED] support for plotly 3.2
- [FIXED] plotly bugfixes for trafo traces and result representation
- [ADDED] Iwamoto algorithm for solving ill-conditioned power flow problems

[1.5.1] - 2018-05-04
----------------------
- [FIXED] delta-wye transformation for 3W-transformers #54
- [ADDED] bus-bus switches collection #76
- [FIXED] some broken documentation links

[1.5.0] - 2018-04-25
----------------------
- [FIXED] plotly hover function for edges (only if use_line_geodata == False)
- [FIXED] from_ppc trafo parameter calculation now also considers baseMVA != 100
- [CHANGED] update create_collection docstrings
- [CHANGED] update HV/MV transformer standard type data
- [ADDED] pp_elements() toolbox function
- [ADDED] new parameter g_us_per_km to model dielectric losses in lines
- [ADDED] single phase short-circuit calculation with negative sequence models
- [ADDED] generic storage model (sgen/load like element with negative / positive power allowed)
- [ADDED] modelling of the complex (voltage magnitude and angle) tap changer for cross control
- [ADDED] modelling of the tap changer of a 3-winding transformer at star point or terminals
- [ADDED] losses of 3W transformers can be modeled at star point, HV, MV or LV side

[1.4.3] - 2018-02-06
----------------------
- [CHANGED] change of collection function names
- [ADDED] sgen collections and ration functionality for sgen and load collections
- [ADDED] cosphi_from_pq toolbox function
- [ADDED] create_nxgraph: respect_switches includes transformer switches

[1.4.2] - 2017-12-05
----------------------
- [ADDED] compatbility with networkx 2.0 (see #82)
- [ADDED] compatibility with pandas 0.21 (see #83)
- [CHANGED] implementation of ZIP loads changed to constant current magnitude paradigm (see #62)
- [ADDED] max_step parameter for shunt
- [ADDED] added warning for large bus index values
- [FIXED] bug in short-circuit results of trafo3w
- [FIXED] bugfix in find_bridges and refactoring
- [CHANGED] faster implementation of result cleanup
- [CHANGED] faster implementation of line index handling in power flow
- [FIXED] bug in plotly label display (#75)
- [ADDED] several fixes, extensions, tests for toolbox
- [ADDED] additional MV line standard types
- [FIXED] kerber extrem vorstadtnetz mv bus voltage
- [FIXED] removed incorrect estimation result tables for load, sgen, gen

[1.4.1] - 2017-09-19
----------------------
- [FIXED] ZIP load issue that led to incorrect calculation of I part with voltage angle shifts
- [FIXED] Bug that set voltage constraints to 0.9/1.2 if no voltage constraints was given in OPF
- [ADDED] possibility to access J matrix after power flow
- [ADDED] opf cost conversion
- [ADDED] opf costs in power system test cases

[1.4.0] - 2017-07-27
----------------------

- [ADDED] possibility to save networks to an sql database
- [CHANGED] major change in fileIO: all networks are converted to a uniform dataframe only version before they are saved as excel, json or sql. Old files can still be loaded, but all files saved with v1.4 can only be loaded with v1.4!
- [FIXED] all tests now pass if numba is not installed (although pandapower might be slow without numba)
- [FIXED] state estimation bug with phase shift transformers
- [CHANGED] OPF now raises specific warning if parameters are missing instead of generic exception
- [ADDED] geographical data for cigre and IEEE case networks
- [ADDED] Dickert LV Networks

[1.3.1] - 2017-06-16
----------------------
- [CHANGED] to_pickle saves only python datatypes and no pickle objects
- [ADDED] html representation of pandapower nets
- [ADDED] collections for trafos, loads, ext_grids
- [CHANGED] renamed create_shunt_as_condensator to create_shunt_as_capacitor
- [FIXED] mock problem in create docstrings
- [ADDED] Synthetic Voltage Control LV Networks

[1.3.0] - 2017-05-10
----------------------
- [ADDED] ZIP loads integrated in power flow
- [ADDED] numba implementation of dissolving switch buses
- [ADDED] Current source representation of full converter elements in short circuit calculations
- [ADDED] Method C for calculation of factor kappa in short circuit calculation
- [CHANGED] Speedup for calculation of branch short circuit currents
- [CHANGED] Branch results for minimum short circuit calculations are calculated as minimal currents
- [ADDED] Interactive plots with plotly
- [CHANGED] included pypower files for power flow and index files
- [FIXED] compatibility with numpy 1.12
- [CHANGED] -1 is a valid value for net.bus_geodata.x
- [CHANGED] allow transformers with negative xk to provide large scale IEEE cases (RTE, PEGASE, Polish)
- [ADDED] large scale IEEE cases (RTE, PEGASE, Polish)
- [ADDED] rated voltage and step variable for shunts
- [ADDED] lagrange multiplier included in bus results after OPF

[1.2.2] - 2017-03-22
--------------------
- [CHANGED] Minor refactoring in pd2ppc
- [ADDED] Technical Report

[1.2.1] - 2017-03-21
--------------------
- [FIXED] Readme for PyPi

[1.2.0] - 2017-03-21
--------------------
- [CHANGED] net.line.imax_ka to net.line.max_i_ka for consistency reasons
- [ADDED] net.line.tp_st_degree for phase shift in trafo tap changers
- [ADDED] sn_kva parameter in create_empty network for per unit system reference power
- [ADDED] parameter parallel for trafo element
- [ADDED] connectivity check for power flow to deal with disconnected network areas
- [ADDED] backward/forward sweep power flow algorithm specially suited for radial and weakly-meshed networks
- [ADDED] linear piece wise and polynomial OPF cost functions
- [ADDED] possibility to make loads controllable in OPF
- [ADDED] to_json and from_json functions to save/load networks with a JSON format
- [ADDED] generator lookup to allow multiple generators at one bus
- [CHANGED] Initialization of calculate_voltage_angles and init for high voltage networks
- [ADDED] bad data detection for state estimation
- [CHANGED] from_ppc: no detect_trafo anymore, several gen at each node possible
- [CHANGED] validate_from_ppc: improved validation behaviour by means of duplicated gen and branch rearangement
- [ADDED] networks: case33bw, case118, case300, case1354pegase, case2869pegase, case9241pegase, GBreducednetwork, GBnetwork, iceland, cigre_network_mv with_der='all' der
- [ADDED] possibility to add fault impedance for short-circuit current calculation
- [ADDED] branch results for short circuits
- [ADDED] static generator model for short circuits
- [ADDED] three winding transformer model for short circuits
- [FIXED] correctly neglecting shunts and tap changer position for short-circuits
- [ADDED] two phase short-circuit current calculation
- [ADDED] tests for short circuit currents with validation against DIgSILENT PowerFactory


[1.1.1] - 2017-01-12
----------------------
- [ADDED] installation description and pypi files from github
- [ADDED] automatic inversion of active power limits in convert format to account for convention change in version 1.1.0
- [CHANGED] install_requires in setup.py


[1.1.0] - 2017-01-11
----------------------
- [ADDED] impedance element can now be used with unsymetric impedances zij != zji
- [ADDED] dcline element that allows modelling DC lines in PF and OPF
- [ADDED] simple plotting function: call pp.simple_plot(net) to directly plot the network
- [ADDED] measurement table for networks. Enables the definition of measurements for real-time simulations.
- [ADDED] estimation module, which provides state estimation functionality with weighted least squares algorithm
- [ADDED] shortcircuit module in beta version for short-circuit calculation according to IEC-60909
- [ADDED] documentation of model validation and tests
- [ADDED] case14, case24_ieee_rts, case39, case57 networks
- [ADDED] mpc and ppc converter
- [CHANGED] convention for active power limits of generators. Generator with max. feed in of 50kW before: p_min_kw=0, p_max_kw=-50. Now p_max_kw=0, p_min_kw=50
- [ADDED] DC power flow function pp.rundcopp
- [FIXED] bug in create_transformer function for tp_pos parameter
- [FIXED] bug in voltage ratio for low voltage side tap changers
- [FIXED] bug in rated voltage calculation for opf line constraints

[1.0.2] - 2016-11-30
----------------------

- [CHANGED] changed in_service dtype from f8 to bool for shunt, ward, xward
- [CHANGED] included i_from_ka and i_to_ka in net.res_line
- [ADDED] recycle parameter added. ppc, Ybus, _is_elements and bus_lookup can be reused between multiple powerflows if recycle["ppc"] == True, ppc values (P,Q,V) only get updated.
- [FIXED] OPF bugfixes: cost scaling, correct calculation of res_bus.p_kw for sgens
- [ADDED] loadcase added as pypower_extension since unnecessary deepcopies were removed
- [CHANGED] supress warnings parameter removed from loadflow, casting warnings are automatically supressed

[1.0.1] - 2016-11-09
----------------------

- [CHANGED] update short introduction example to include transformer
- [CHANGED] included pypower in setup.py requirements (only pypower, not numpy, scipy etc.)
- [CHANGED] mpc / ppc renamed to ppci / ppc
- [FIXED] MANIFEST.ini includes all relevant doc files and exclude report
- [FIXED] handling of tp_pos parameter in create_trafo and create_trafo3w
- [FIXED] init="result" for open bus-line switches
