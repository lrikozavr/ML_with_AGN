#!/home/kiril/python_env_iron_ment/new_proj/bin/python

from dustmaps.config import config
config['data_dir'] = '/media/kiril/j_08/dust_map/'

import dustmaps.sfd
dustmaps.sfd.fetch()

import dustmaps.planck
dustmaps.planck.fetch()

import dustmaps.planck
dustmaps.planck.fetch(which='GNILC')

import dustmaps.bayestar
dustmaps.bayestar.fetch()

import dustmaps.iphas
dustmaps.iphas.fetch()

import dustmaps.marshall
dustmaps.marshall.fetch()

import dustmaps.chen2014
dustmaps.chen2014.fetch()

import dustmaps.lenz2017
dustmaps.lenz2017.fetch()

import dustmaps.pg2010
dustmaps.pg2010.fetch()

import dustmaps.leike_ensslin_2019
dustmaps.leike_ensslin_2019.fetch()

import dustmaps.leike2020
dustmaps.leike2020.fetch()