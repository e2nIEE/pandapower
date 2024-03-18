import logging
import time

import numpy
import pandas as pd

from pandapower.converter.cim import cim_tools
from pandapower.converter.cim.cim2pp import build_pp_net
from pandapower.converter.cim.other_classes import Report, LogLevel, ReportCode

logger = logging.getLogger('cim.cim2pp.converter_classes.geoCoordinatesFromGLCim16')

sc = cim_tools.get_pp_net_special_columns_dict()


class GeoCoordinatesFromGLCim16:
    def __init__(self, cimConverter: build_pp_net.CimConverter):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cimConverter = cimConverter

    def add_geo_coordinates_from_gl_cim16(self):
        # TODO: check if crs is WGS84 -> if not, convert to WGS84
        self.logger.info("Creating the geo coordinates from CGMES GeographicalLocation.")
        time_start = time.time()

        # create dataframe with columns 'xPosition', 'yPosition', 'sequenceNumber', 'PowerSystemResources'(rdfId)
        gl_data = pd.merge(
            self.cimConverter.cim['gl']['PositionPoint'][['Location', 'xPosition', 'yPosition', 'sequenceNumber']],
            self.cimConverter.cim['gl']['Location'][['rdfId', 'PowerSystemResources']], how='left',
            left_on='Location', right_on='rdfId')
        gl_data.drop(columns=['Location', 'rdfId'], inplace=True)
        # make sure that the columns 'xPosition' and 'yPosition' are floats
        gl_data['xPosition'] = gl_data['xPosition'].astype(float)
        gl_data['yPosition'] = gl_data['yPosition'].astype(float)
        bus_geo = gl_data.rename(columns={'PowerSystemResources': 'Substation'})

        # create lookup table with columns `origin_id` and `Substation`
        cn = self.cimConverter.cim['eq']['ConnectivityNode'][['rdfId', 'ConnectivityNodeContainer']]
        cn = pd.concat([cn, self.cimConverter.cim['eq_bd']['ConnectivityNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn = pd.concat([cn, self.cimConverter.cim['tp']['TopologicalNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn = pd.concat([cn, self.cimConverter.cim['tp_bd']['TopologicalNode'][['rdfId', 'ConnectivityNodeContainer']]])
        cn.rename(columns={'rdfId': sc['o_id'], 'ConnectivityNodeContainer': 'rdfId'}, inplace=True)
        cn = pd.merge(cn, self.cimConverter.cim['eq']['VoltageLevel'][['rdfId', 'Substation']], how='left', on='rdfId')
        cn.drop(columns=['rdfId'], inplace=True)

        buses = self.cimConverter.net.bus.reset_index()
        buses = buses[['index', sc['o_id']]]
        buses = pd.merge(buses, cn, how='left', on=sc['o_id'])

        bus_geo = pd.merge(bus_geo, buses, how='inner', on='Substation')
        bus_geo.drop(columns=['Substation'], inplace=True)
        bus_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        bus_geo['coords'] = bus_geo[['xPosition', 'yPosition']].values.tolist()
        bus_geo['coords'] = bus_geo[['coords']].values.tolist()
        bus_geo['geo'] = bus_geo.apply(
            lambda row: f'{{"coordinates": [{row["xPosition"]}, {row["yPosition"]}], "type": "Point"}}', axis=1
        )
        # for the buses which have more than one coordinate
        bus_geo_mult = bus_geo[bus_geo.duplicated(subset=sc['o_id'], keep=False)]
        # now deal with the buses which have more than one coordinate
        for _, df_group in bus_geo_mult.groupby(by=sc['o_id'], sort=False):
            bus_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
            bus_geo['geo'][df_group.index.values[
                0]] = f'{{"coordinates": {df_group["coords"][df_group.index.values[0]]}, "type": "LineString"}}'
        bus_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        bus_geo.sort_values(by='index', inplace=True)
        self.cimConverter.net.bus.geo = bus_geo.copy().set_index('index').geo

        # the geo coordinates for the lines
        lines = self.cimConverter.net.line.reset_index()
        lines = lines[['index', sc['o_id']]]
        line_geo = gl_data.rename(columns={'PowerSystemResources': sc['o_id']})
        line_geo = pd.merge(line_geo, lines, how='inner', on=sc['o_id'])
        line_geo.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
        line_geo['coords'] = line_geo[['xPosition', 'yPosition']].values.tolist()
        line_geo['coords'] = line_geo[['coords']].values.tolist()
        for _, df_group in line_geo.groupby(by=sc['o_id']):
            line_geo['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
        line_geo.drop_duplicates([sc['o_id']], keep='first', inplace=True)
        line_geo['geo'] = line_geo.apply(lambda row: f'{{"coordinates": {row["coords"]}, "type": "LineString"}}',
                                         axis=1)
        # now add the line coordinates
        self.cimConverter.net.line.geo = line_geo.copy().set_index('index').geo

        # now create geo coordinates which are official not supported by pandapower, e.g. for transformer
        for one_ele in ['trafo', 'trafo3w', 'switch', 'ext_grid', 'load', 'sgen', 'gen', 'impedance', 'dcline', 'shunt',
                        'storage', 'ward', 'xward']:
            one_ele_df = self.cimConverter.net[one_ele][[sc['o_id']]]
            one_ele_df = pd.merge(gl_data.rename(columns={'PowerSystemResources': sc['o_id']}),
                                  one_ele_df, how='inner', on=sc['o_id'])
            one_ele_df.sort_values(by=[sc['o_id'], 'sequenceNumber'], inplace=True)
            one_ele_df['coords'] = one_ele_df[['xPosition', 'yPosition']].values.tolist()
            one_ele_df['coords'] = one_ele_df[['coords']].values.tolist()
            for _, df_group in one_ele_df.groupby(by=sc['o_id']):
                one_ele_df['coords'][df_group.index.values[0]] = df_group[['xPosition', 'yPosition']].values.tolist()
            one_ele_df.drop_duplicates([sc['o_id']], keep='first', inplace=True)
            # now add the coordinates
            self.cimConverter.net[one_ele]['coords'] = self.cimConverter.net[one_ele][sc['o_id']].map(
                one_ele_df.set_index(sc['o_id']).to_dict(orient='dict').get('coords'))

        self.logger.info("Finished creating the GL coordinates, needed time: %ss" % (time.time() - time_start))
        self.cimConverter.report_container.add_log(Report(
            level=LogLevel.INFO, code=ReportCode.INFO_CONVERTING,
            message="Created the GL coordinates, needed time: %ss" % (time.time() - time_start)))
