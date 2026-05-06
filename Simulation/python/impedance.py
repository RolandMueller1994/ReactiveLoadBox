from pathlib import Path
from typing import Tuple, Union
import numpy as np
import yaml
import re
import matplotlib.pyplot as plt
import math

from prettytable import PrettyTable


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


def plotPower(p: list, pInput, cfgPath: Union[Path, None]=None):
    if len(p) == 0:
        return

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 5))
    fLst = [pPt['f'] for pPt in p]
    for key, label in [('p_total', 'Total'),
                       ('p_ser', 'Main R'),
                       ('p_res_r', 'Resonance R'),
                       ('p_res_c', 'Resonance C'),
                       ('p_res_l', 'Resonance L'),
                       ('p_ramp_r', 'Ramp 1 R'),
                       ('p_ramp_l', 'Ramp 1 L'),
                       ('p_ramp2_r', 'Ramp 2 R'),
                       ('p_ramp2_l', 'Ramp 2 L'),
                       ]:
        if key not in p[0]:
            continue
        pLst = [abs(pPt[key].real) for pPt in p]
        ax.plot(fLst, pLst, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [W]')
    plt.legend()
    plt.tight_layout()
    if cfgPath is not None:
        figPath = Path('fig') / (str(cfgPath.stem) + '_power.png')
    else:
        figPath = Path('fig/power.png')
    plt.savefig(figPath, dpi=300)
    plt.show()

    pDb = [10 * math.log10(abs(pPt['p_total'].real)/pInput) for pPt in p]
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 5))
    ax.plot(fLst, pDb)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.set_xscale('log')
    plt.tight_layout()
    if cfgPath is not None:
        figPath = Path('fig') / (str(cfgPath.stem) + '_power_total.png')
    else:
        figPath = Path('fig/power_total.png')
    plt.savefig(figPath, dpi=300)
    plt.show()


def calcImpedancePt(freq, r_ser, r_main, c_main, l_main, l_main_r, r_ramp1, l_ramp1, l_ramp1_r, r_ramp2=None, l_ramp2=None,
                    l_ramp2_r=None, i_i=None, r_drv=None) -> Tuple[float, float, Union[dict, None]]:
    w = 2 * math.pi * freq
    w = w.item()

    z_c_main = 1 / (w*c_main*1j)
    z_l_main = w*l_main*1j + l_main_r
    z_res = 1 / (1/z_c_main + 1/z_l_main)

    z_main = 1 / (1/r_main + 1/z_res)
    z_l_ramp = w*l_ramp1*1j + l_ramp1_r
    z_ramp = 1 / (1/r_ramp1 + 1/z_l_ramp)

    if r_ramp2 is not None:
        z_l_ramp2 = w*l_ramp2*1j + l_ramp2_r
        z_ramp2 = 1 / (1/r_ramp2 + 1 / z_l_ramp2)
    else:
        z_ramp2 = 0
        z_l_ramp2 = 0

    z = r_ser + z_main + z_ramp + z_ramp2

    if i_i is not None:
        i = i_i * r_drv / (r_drv + z)
        p_ser = abs(i)**2 * r_ser

        v_res = i * z_main
        p_res_r = abs(v_res)**2 / r_main
        p_res_c = abs(v_res)**2 / z_c_main
        p_res_l = abs(v_res)**2 / z_l_main

        v_ramp = i * z_ramp
        p_ramp_r = abs(v_ramp)**2 / r_ramp1
        p_ramp_l = abs(v_ramp)**2 / z_l_ramp

        p_dict = {
            'p_total': i**2*z,
            'p_ser': p_ser,
            'p_res_r': p_res_r,
            'p_res_c': p_res_c,
            'p_res_l': p_res_l,
            'p_ramp_r': p_ramp_r,
            'p_ramp_l': p_ramp_l,
            'f': freq
        }

        if r_ramp2 is not None:
            v_ramp2 = i * z_ramp2
            p_ramp2_r = v_ramp2**2 / r_ramp2
            p_ramp2_l = v_ramp2**2 / z_l_ramp2
            p_dict['p_ramp2_r'] = p_ramp2_r
            p_dict['p_ramp2_l'] = p_ramp2_l

    else:
        p_dict = None

    phi = math.atan2(z.imag, z.real)
    return abs(z), phi/math.pi * 180, p_dict

def calcImpedance(freq: np.ndarray, circuitInfo: dict, drv_info: dict) -> Tuple[np.ndarray, np.ndarray, list]:
    mag = np.zeros_like(freq)
    phi = np.zeros_like(freq)
    p = []

    if 'p_drv' in drv_info:
        r_drv = drv_info['r_drv']
        v_i = math.sqrt(drv_info['p_drv'] * r_drv)
        i_i = v_i / (1 / (1/r_drv + 1/drv_info['r_int']))
    else:
        r_drv = None
        i_i = None

    for cnt, f in enumerate(freq):
        magPt, phiPt, p_dict = calcImpedancePt(f, i_i=i_i, r_drv=drv_info['r_int'],  **circuitInfo)
        mag[cnt] = magPt
        phi[cnt] = phiPt
        if p_dict is not None:
            p.append(p_dict)

    return mag, phi, p

def analyzeImpedance(cfgPath: Path) -> None:
    with open(cfgPath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
    measPath = Path(cfg['meas_config']['path'])

    fMeas, impMeas, phiMeas = readLimpMeasurement(measPath)
    impSim, phiSim, p = calcImpedance(fMeas, cfg['circuit_info'], cfg.get('drv_info', dict()))

    rSim = np.abs(impSim)
    rMeas = np.abs(impMeas)
    for cnt, f in enumerate(fMeas):
        if f > 300:
            break

    rMaxIdx = np.argmax(rSim[:cnt])
    rMax = rSim[rMaxIdx]
    fMax = fMeas[rMaxIdx]

    rMaxMeasIdx = np.argmax(rMeas[:cnt])
    rMaxMeas = rMeas[rMaxMeasIdx]
    fMaxMeas = fMeas[rMaxMeasIdx]

    table = PrettyTable(['Type', 'Value'])
    table.add_divider()
    table.add_row(['f Res. Meas.', f'{fMaxMeas:.2f}'])
    table.add_row(['R Res. Meas.', f'{rMaxMeas:.2f}'])
    table.add_divider()
    table.add_row(['f Res. Sim.', f'{fMax:.2f}'])
    table.add_row(['R Res. Sim.', f'{rMax:.2f}'])
    table.add_divider()

    pCfg = cfg.get('drv_info', None)
    if pCfg is not None:
        pTarget = pCfg['p_drv']
        pTotal = np.asarray([10 * math.log10(abs(pPt['p_total']) / pTarget) for pPt in p])
        pMaxIdx = np.argmax(pTotal[:cnt])
        pMax = pTotal[pMaxIdx]
        pMaxF = fMeas[pMaxIdx]
        table.add_row(['f P Max', f'{pMaxF:.2f}'])
        table.add_row(['P Max', f'{pMax:.2f}'])
        table.add_divider()

    print(table)

    plotImpedance(fMeas, np.vstack((impMeas, impSim)), np.vstack((phiMeas, phiSim)), labels=['Meas.', 'Sim.'], cfgPath=cfgPath)
    plotPower(p, cfg.get('drv_info', {}).get('p_drv', None), cfgPath=cfgPath)


if __name__ == '__main__':
    cfgPath = Path('cfg/config_1.yaml')
    analyzeImpedance(cfgPath)