"""
Generates the g4beamline file for the whole setup containing all the setup parameters
"""
from measurements import (converter, half_block, collimator, separator,
                          kapton_tape,virtual_detector, z_positions)

PREAMBLE = ("# The 'default' physics list is QGSP_BERT\n"
                    "physics QGSP")
CONVERTER = (
    f"#Bismuth converters\n"
    f"box bismuth_converter material={converter['material']} "
    f"width={converter['width']} height={converter['height']} "
    f"length={converter['length']} "
    f"color={converter['color']}\n"
    f"place bismuth_converter z={z_positions['converter']}"
    )
HALF_BLOCK = (
    f"#Tungsten half block\n"
    f"box tungsten_block material={half_block['material']} "
    f"width={half_block['width']} height={half_block['height']} "
    f"length={half_block['length']}\n"
    f"place tungsten_block x=0 y=-{half_block['height']/2} z={z_positions['half block']} "
    f"kill={half_block['kill']}"
)
COLLIMATOR = (
    f"#Talantum collimator\n"
    f"cylinder talantum_col material={collimator['material']} "
    f"innerRadius={collimator['inner radius']} outerRadius={collimator['outer radius']} "
    f"length={collimator['length']}\n"
    f"place talantum_col z={z_positions['collimator']}"
)
SEPARATOR = (
    f"#Separator magnet to remove Bethe-Heitler\n"
    f"genericbend separator By={separator['By']} fieldMaterial=Vacuum "
    f"fieldWidth={separator['width']} fieldHeight={separator['height']} "
    f"fieldLength={separator['length']} "
    f"fieldColor={separator['color']}\n"
    f"place separator z={z_positions['separator']}"
)

KAPTON_TAPE = (
    f"#Kapton Tape\n"
    f"param d=1 cos=0.76604444311 sin=0.642787609 angle=40\n"
    f"box kapton_tape material={kapton_tape['material']} "
    f"width={kapton_tape['width']} height={kapton_tape['height']} "
    f"length={kapton_tape['length']} "
    f"color={kapton_tape['color']}\n"
    f"place kapton_tape x=0 y=-$d-{kapton_tape['length']/2}*$sin "
    f"z={z_positions['kapton tape']}-{kapton_tape['length']/2}*$cos "
    f"rotation=x-$angle"
)

VIRTUAL_DETECTOR = (
    f"#Virtual detector\n"
    f"virtualdetector noise_measure_Det filename={virtual_detector['filename']} "
    f"format='ascii' color={virtual_detector['color']} "
    f"width={virtual_detector['width']} height={virtual_detector['height']} "
    f"length={virtual_detector['length']}\n"
    f"place noise_measure_Det z={z_positions['virtual detector']}"
)

VIRTUAL_DETECTOR_GAMMAPROFILE = (
    f"#Virtual detector\n"
    f"virtualdetector noise_measure_Det filename={virtual_detector['filename']} "
    f"format='ascii' color={virtual_detector['color']} "
    f"width={virtual_detector['width']} height={virtual_detector['height']} "
    f"length={virtual_detector['length']}\n"
    f"place noise_measure_Det z={z_positions['virtual detector gamma']}"
)

MEASUREMENTS = (
    f"#Measuring boxes\n"
    f"cylinder origin outerRadius=100 length=0.01 material=Vacuum color=0,0,1,0.3\n"
    f"place origin z=0\n"
    f"cylinder measure_converter outerRadius=100 length=0.01 material=Vacuum color=0,0,1,0.3\n"
    f"place measure_converter z={z_positions['converter']}\n"
    f"cylinder measure_half_block outerRadius=100 length=0.01 material=Vacuum color=0,0,1,0.3\n"
    f"place measure_half_block z={z_positions['half block']}\n"
    f"cylinder measure_collimator outerRadius=100 length=0.01 material=Vacuum color=0,0,1,0.3\n"
    f"place measure_collimator z={z_positions['collimator']}\n"
    f"cylinder measure_separator outerRadius=100 length=0.01 material=Vacuum color=0,0,1,0.3\n"
    f"place measure_separator z={z_positions['separator']}\n"
    f"cylinder measure_kapton outerRadius=100 length=0.01 material=Vacuum color=0,0,1,0.3\n"
    f"place measure_kapton z={z_positions['kapton tape']}\n"
    f"cylinder measure_detector outerRadius=100 length=0.01 material=Vacuum color=0,0,1,0.3\n"
    f"place measure_detector z={z_positions['virtual detector']}\n"
    f"box measure_axis length=2000 width=100 height=0.01 material=Vacuum color=0,0,1,0.3\n"
    f"place measure_axis x=50 y=0 z=1000\n"
    f"box measure_kapton_d length=2000 width=100 height=0.01 material=Vacuum color=1,0,1,0.3\n"
    f"place measure_kapton_d x=50 y=-$d z=1000\n"
)

TEST_BEAM = (
    f"#Particle beams\n"
    f"beam gaussian particle=gamma nEvents=100 beamZ={z_positions['converter']} "
    f"meanMomentum=1000"
)

def gen_setup(
        g4blfilename: str,
        beams: str=None,
        measure=False,
        test_beam=False,
        measure_beam_profile=False,
        write_file=True
        ) -> str:
    """Writes the setup script for g4beamline

    Args:
        g4blfilename (str): the g4beamline file name (without extension)
        beams (str, optional): g4beamline beams to import into setup. Defaults to None.
        measure (bool, optional): True to place measurement planes. Defaults to False.
        test_beam (bool, optional): True to place test beam. Defaults to False.
        write_file (bool, optional): True to write the output to a .g4bl file. Defaults to True.

    Returns:
        str: The string for the g4beamline setup.
    """
    if measure:
        measure_string = MEASUREMENTS
    else:
        measure_string=""

    if test_beam:
        beam_string = TEST_BEAM
    else:
        beam_string = "#Particle beams"

    if beams is not None:
        beam_string = beams
        if test_beam:
            print('Beam import detected. Removing test beam.')
    
    if measure_beam_profile:
        kapton_tape_string = ""
        virtual_detector_string = VIRTUAL_DETECTOR_GAMMAPROFILE
    else:
        kapton_tape_string = KAPTON_TAPE
        virtual_detector_string = VIRTUAL_DETECTOR

    write_string = (
        f"{PREAMBLE}\n\n"
        f"{beam_string}\n\n"
        f"{CONVERTER}\n\n"
        f"{HALF_BLOCK}\n\n"
        f"{COLLIMATOR}\n\n"
        f"{SEPARATOR}\n\n"
        f"{kapton_tape_string}\n\n"
        f"{virtual_detector_string}\n"
        f"\n{measure_string}\n"
    )

    if write_file:
        with open(f"{g4blfilename}.g4bl", "w", encoding="utf-8") as file_handler:
            file_handler.write(write_string)

    return write_string

if __name__ == '__main__':
    gen_setup('setup_test', measure=False, test_beam=False, measure_beam_profile=True)
