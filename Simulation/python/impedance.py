from pathlib import Path
from typing import Tuple, Union
import numpy as np
import yaml
import re
import matplotlib.pyplot as plt
import math


def readLimpMeasurement(fPath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pattern = re.compile(r'(\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)')
    fLst = []
    magLst = []
    phiLst = []
    with open(fPath, 'r') as f:
        for line in f:
            line = line.strip()
            match = re.match(pattern, line)
            if match:
                f = float(match.group(1))
                mag = float(match.group(2))
                phi = float(match.group(3))
                fLst.append(f)
                magLst.append(mag)
                phiLst.append(phi)
    return np.asarray(fLst), np.asarray(magLst), np.asarray(phiLst)


def plotImpedance(freq: np.ndarray, mag: np.ndarray, phase: np.ndarray, labels: list=None, cfgPath: Union[Path, None]=None) -> None:
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 10))
    ax1 = ax[0]
    ax2 = ax[1]

    if len(mag.shape) == 2:
        if labels is not None:
            label1, label2 = labels
        else:
            label1 = label2 = None
        ax1.plot(freq, mag[0], label=label1)
        ax1.plot(freq, mag[1], label=label2)

        ax2.plot(freq, phase[0], label=label1)
        ax2.plot(freq, phase[1], label=label2)
    else:
        ax1.plot(freq, mag, label=labels)
        ax1.plot(freq, phase, label=labels)
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Impedance [Ohm]')
    ax1.set_xscale('log')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Phase [deg]')
    ax2.set_xscale('log')

    if labels is not None:
        ax1.legend()
        ax2.legend()

    if cfgPath is not None:
        figPath = Path('fig') / (str(cfgPath.stem) + '_impedance.png')
    else:
        figPath = Path('fig/impedance.png')
    plt.tight_layout()
    plt.savefig(figPath, dpi=300)
    plt.show(block=True)


def calcImpedancePt(freq, r_ser, r_main, c_main, l_main, l_main_r, r_ramp1, l_ramp1, l_ramp1_r, r_ramp2=None, l_ramp2=None,
                    l_ramp2_r=None) -> Tuple[float, float]:
    w = 2 * math.pi * freq
    w = w.item()

    z_main = 1 / (1/r_main + 1/(w*l_main*1j + l_main_r) + w*c_main*1j)
    z_ramp = 1 / (1/r_ramp1 + 1/(w*l_ramp1*1j + l_ramp1_r))

    if r_ramp2 is not None:
        z_ramp2 = 1 / (1/r_ramp2 + 1 / (w*l_ramp2*1j + l_ramp2_r))
    else:
        z_ramp2 = 0

    z = r_ser + z_main + z_ramp + z_ramp2
    phi = math.atan2(z.imag, z.real)
    return abs(z), phi/math.pi * 180

def calcImpedance(freq: np.ndarray, circuitInfo: dict) -> Tuple[np.ndarray, np.ndarray]:
    mag = np.zeros_like(freq)
    phi = np.zeros_like(freq)

    for cnt, f in enumerate(freq):
        magPt, phiPt = calcImpedancePt(f, **circuitInfo)
        mag[cnt] = magPt
        phi[cnt] = phiPt

    return mag, phi

def analyzeImpedance(cfgPath: Path) -> None:
    with open(cfgPath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
    measPath = Path(cfg['meas_config']['path'])

    fMeas, impMeas, phiMeas = readLimpMeasurement(measPath)
    impSim, phiSim = calcImpedance(fMeas, cfg['circuit_info'])

    plotImpedance(fMeas, np.vstack((impMeas, impSim)), np.vstack((phiMeas, phiSim)), labels=['Meas.', 'Sim.'], cfgPath=cfgPath)


if __name__ == '__main__':
    cfgPath = Path('cfg/config_1.yaml')
    analyzeImpedance(cfgPath)