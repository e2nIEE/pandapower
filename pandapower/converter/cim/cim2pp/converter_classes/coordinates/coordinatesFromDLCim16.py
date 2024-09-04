import logging
import time

import numpy
import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.geoCoordinatesFromDLCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class CoordinatesFromDLCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def add_coordinates_from_dl_cim16(self, diagram_name: str = None):
        self.logger.info("Creating the coordinates from CGMES DiagramLayout.")
        time_start = time.time()
        # choose a diagram if it is not given (the first one ascending)
        if diagram_name is None:
            diagram_name = self.cimConverter.cim['dl']['Diagram'].sort_values(by='name')['name'].values[0]
        self.logger.debug("Choosing the geo coordinates from diagram %s" % diagram_name)
        if diagram_name != 'all':
            # reduce the source data to the chosen diagram only
            diagram_rdf_id = \
                self.cimConverter.cim['dl']['Diagram']['rdfId'][
                    self.cimConverter.cim['dl']['Diagram']['name'] == diagram_name].values[0]
            dl_do = self.cimConverter.cim['dl']['DiagramObject'][
                self.cimConverter.cim['dl']['DiagramObject']['Diagram'] == diagram_rdf_id]
            dl_do = dl_do.rename(columns={'rdfId': 'DiagramObject'})
        else:
            dl_do = self.cimConverter.cim['dl']['DiagramObject'].copy()
            dl_do = dl_do.rename(columns={'rdfId': 'DiagramObject'})
        dl_data = pd.merge(dl_do, self.cimConverter.cim['dl']['DiagramObjectPoint'], how='left', on='DiagramObject')
        dl_data = dl_data.drop(columns=['rdfId', 'Diagram', 'DiagramObject'])
        # make sure that the columns 'xPosition' and 'yPosition' are floats
        dl_data['xPosition'] = dl_data['xPosition'].astype(float)
        dl_data['yPosition'] = dl_data['yPosition'].astype(float)
        dl_data['coords_str'] = '['+dl_data['xPosition'].astype(str)+', '+dl_data['yPosition'].astype(str)+']'

        # create coordinates for the different assets bus, line, transformer
        for one_ele in ['bus', 'trafo', 'trafo3w', 'switch', 'ext_grid', 'load', 'sgen', 'gen', 'line', 'dcline',
                        'impedance', 'shunt', 'storage', 'ward', 'xward']:
            one_ele_df = self.cimConverter.net[one_ele][[sc['o_id']]]
            one_ele_df = pd.merge(dl_data, one_ele_df, how='inner', left_on='IdentifiedObject', right_on=sc['o_id'])
            one_ele_df = one_ele_df.sort_values(by=[sc['o_id'], 'sequenceNumber'])
            if one_ele != 'line' and one_ele != 'dcline' and one_ele != 'impedance':
                # only one coordinate for each asset (except line and impedance)
                one_ele_df = one_ele_df.drop_duplicates([sc['o_id']], keep='first')
                one_ele_df['diagram'] = '{"coordinates": ' + one_ele_df["coords_str"] + ', "type": "Point"}'
            else:
                # line strings
                one_ele_df['coords'] = one_ele_df[['xPosition', 'yPosition']].values.tolist()
                one_ele_df['coords'] = one_ele_df[['coords']].values.tolist()
                for _, df_group in one_ele_df.groupby(by=sc['o_id']):
                    one_ele_df.at[df_group.index.values[0], 'coords'] = df_group[
                        ['xPosition', 'yPosition']].values.tolist()
                one_ele_df = one_ele_df.drop_duplicates([sc['o_id']], keep='first')
                one_ele_df['coords'] = one_ele_df['coords'].astype(str)
                one_ele_df['diagram'] = '{"coordinates": '+one_ele_df['coords'].astype(str)+', "type": "LineString"}'

            self.cimConverter.net[one_ele]['diagram'] = self.cimConverter.net[one_ele][sc['o_id']].map(
                one_ele_df.set_index(sc['o_id']).to_dict(orient='dict').get('diagram'))

        self.logger.info("Finished creating the DL coordinates, needed time: %ss" % (time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created the DL coordinates, needed time: %ss" % (time.time() - time_start)))
