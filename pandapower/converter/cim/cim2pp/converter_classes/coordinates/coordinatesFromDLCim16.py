import logging
import time

import geojson
import numpy as np
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
            dl_do.rename(columns={'rdfId': 'DiagramObject'}, inplace=True)
        else:
            dl_do = self.cimConverter.cim['dl']['DiagramObject'].copy()
            dl_do.rename(columns={'rdfId': 'DiagramObject'}, inplace=True)
        dl_data = pd.merge(dl_do, self.cimConverter.cim['dl']['DiagramObjectPoint'], how='left', on='DiagramObject')
        dl_data.drop(columns=['rdfId', 'Diagram', 'DiagramObject'], inplace=True)
        # make sure that the columns 'xPosition' and 'yPosition' are floats
        dl_data['xPosition'] = dl_data['xPosition'].astype(float)
        dl_data['yPosition'] = dl_data['yPosition'].astype(float)
        # the coordinates for the buses
        buses = self.cimConverter.net.bus.reset_index()
        buses = buses[['index', sc['o_id']]]
        bus_geo = pd.merge(dl_data, buses, how='inner', left_on='IdentifiedObject', right_on=sc['o_id'])
        bus_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        bus_geo['coords'] = bus_geo[['xPosition', 'yPosition']].values.tolist()
        bus_geo['coords'] = bus_geo[['coords']].values.tolist()
        # for the buses which have more than one coordinate
        bus_geo_mult = bus_geo[bus_geo.duplicated(subset=sc['o_id'], keep=False)]
        # now deal with the buses which have more than one coordinate
        for _, df_group in bus_geo_mult.groupby(by=sc['o_id'], sort=False):
            bus_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        bus_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        bus_geo.sort_values(by='index', inplace=True)
        self.cimConverter.net.bus['diagram'] = bus_geo.copy().set_index('index').coords

        # the coordinates for the lines
        lines = self.cimConverter.net.line.reset_index()
        lines = lines[['index', sc['o_id']]]
        line_geo = pd.merge(dl_data, lines, how='inner', left_on='IdentifiedObject', right_on=sc['o_id'])
        line_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        line_geo['coords'] = line_geo[['xPosition', 'yPosition']].values.tolist()
        line_geo['coords'] = line_geo[['coords']].values.tolist()
        for _, df_group in line_geo.groupby(by=sc['o_id']):
            line_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        line_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        line_geo.sort_values(by='index', inplace=True)
        # now add the line coordinates
        self.cimConverter.net.line['diagram'] = line_geo.copy().set_index('index').coords

        # now create coordinates which are official not supported by pandapower, e.g. for transformer
        for one_ele in ['trafo', 'trafo3w', 'switch', 'ext_grid', 'load', 'sgen', 'gen', 'impedance', 'dcline', 'shunt',
                        'storage', 'ward', 'xward']:
            one_ele_df = self.cimConverter.net[one_ele][[sc['o_id']]]
            one_ele_df = pd.merge(dl_data, one_ele_df, how='inner', left_on='IdentifiedObject', right_on=sc['o_id'])
            one_ele_df.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
            one_ele_df['coords'] = one_ele_df[['xPosition', 'yPosition']].values.tolist()
            one_ele_df['coords'] = one_ele_df[['coords']].values.tolist()
            for _, df_group in one_ele_df.groupby(by=sc['o_id']):
                one_ele_df['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
            one_ele_df.drop_duplicates([sc['o_id']], keep='first', inplace=True)
            # now add the coordinates
            self.cimConverter.net[one_ele]['coords'] = self.cimConverter.net[one_ele][sc['o_id']].map(
                one_ele_df.set_index(sc['o_id']).to_dict(orient='dict').get('coords'))

        self.logger.info("Finished creating the DL coordinates, needed time: %ss" % (time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created the DL coordinates, needed time: %ss" % (time.time() - time_start)))
