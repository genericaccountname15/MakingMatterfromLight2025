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

def gen_setup(g4blfilename: str):
    """Writes the setup script for g4beamline

    Args:
        g4blfilename (str): the g4beamline file name (without extension)
    """
    with open(f"{g4blfilename}.g4bl", "w", encoding="utf-8") as file_handler:
        write_string = (
            f"{PREAMBLE}\n\n"
            f"#Particle beams\n\n"
            f"{CONVERTER}\n\n"
            f"{HALF_BLOCK}\n\n"
            f"{COLLIMATOR}\n\n"
            f"{SEPARATOR}\n\n"
            f"{KAPTON_TAPE}\n\n"
            f"{VIRTUAL_DETECTOR}\n"
        )
        file_handler.write(write_string)

if __name__ == '__main__':
    gen_setup('whole_setup')
