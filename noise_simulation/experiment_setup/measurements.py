"""
Measurements of the laser plasma platform provided by Brendan Kettle
Stored in python dicts
Measurements are in units of mm

Timothy Chew
11/8/25
"""
converter = {
    'width': 8,
    'height': 8,
    'length': 1,
    'material': 'Bi',
    'color': '1,0.84,0'
}

half_block = {
    'width': 20,
    'height': 50,
    'length': 50,
    'material': 'W',
    'kill': 1
}

collimator = {
    'outer radius': 10,
    'inner radius': 1,
    'length': 100,
    'material': 'Ta',
    'kill': 1
}

separator = {
    'width': 100,
    'height': 200,
    'length': 400,
    'By': 3,    #T
    'color': '1,0,0,0.3',

}

kapton_tape = {
    'width': 8,
    'height': 0.025,
    'length': 25,
    'material': 'KAPTON',
    'color': '1,0.5,0'
}

virtual_detector = {
    'filename': 'noise_measure_Det.txt',
    'width': 100,
    'height': 100,
    'length': 0.001,
    'color': '0,1,0,0.3'
}

z_positions = {
    'converter': 194,
    'half block': 195 + 85 + half_block['length']/2,
    'collimator': 195 + 85 + half_block['length'] + 45 + collimator['length']/2,
    'separator': 195 + 85 + half_block['length'] + 45
                + collimator['length'] + 25 + separator['length']/2,
    'kapton tape': 195 + 85 + half_block['length'] + 45 + collimator['length']
                + 25 + separator['length'] + 160,
    'virtual detector': 195 + 85 + half_block['length'] + 45 + collimator['length']
                + 25 + separator['length'] + 160 + 360,
    'virtual detector gamma': 195 + 85 + half_block['length'] + 45 + collimator['length']
                + 25 + separator['length'] + 160,
}
