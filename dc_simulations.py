#!/bin/python3
"""
Code to simulate a simple direct converter receiver to compute the following parameters:
- Maximum noise figure of the line up
- CNR at the output of the receiver
"""
import numpy as np


class Duplexer:
    def __init__(self, gain=-1, NF=1, Nthin=False):
        self.gain = gain
        self.NF = NF
        self.Nthin = Nthin

    def __call__(self, Pin, Nin):
        return {
            "Pout": Pin + self.gain,
            "Nout": Nin + (self.NF if self.Nthin else 0) + self.gain,
            "IMD2": None,
            "IMD3": None,
        }


class LNA:
    def __init__(self, gain=20, NF=3, IIP2=0, IIP3=29, P1dB=10):
        self.gain = gain
        self.NF = NF
        self.IIP2 = IIP2
        self.IIP3 = IIP3
        self.P1dB = P1dB

    def __call__(self, Pin, Nin):
        return {
            "Pout": Pin + self.gain,
            "Nout": Nin + self.NF + self.gain,
            "IMD2": (2 * Pin - self.IIP2) + self.gain,
            "IMD3": (3 * Pin - (2 * self.IIP3)) + self.gain,
        }


class BPF:
    def __init__(self, gain=-1, NF=1, Nthin=False):
        self.gain = gain
        self.NF = NF
        self.Nthin = Nthin

    def __call__(self, Pin, Nin):
        return {
            "Pout": Pin + self.gain,
            "Nout": Nin + (self.NF if self.Nthin else 0) + self.gain,
            "IMD2": None,
            "IMD3": None,
        }


class VGA:
    def __init__(self, gain=20, NF=3, IIP2=0, IIP3=29, P1dB=10):
        self.gain = gain
        self.NF = NF
        self.IIP2 = IIP2
        self.IIP3 = IIP3
        self.P1dB = P1dB

    def __call__(self, Pin, Nin):
        return {
            "Pout": Pin + self.gain,
            "Nout": Nin + self.NF + self.gain,
            "IMD2": (2 * Pin - self.IIP2) + self.gain,
            "IMD3": (3 * Pin - (2 * self.IIP3)) + self.gain,
        }


class QDEMOD:
    def __init__(self, gain=0, NF=1, IIP2=0, IIP3=29, IS=40):
        self.gain = gain
        self.NF = NF
        self.IIP2 = IIP2
        self.IIP3 = IIP3
        self.IS = IS  # isolation gain between LO and BB

    def __call__(self, Pin, Nin):
        return {
            "Pout": Pin + self.gain,
            "Nout": Nin + self.NF + self.gain,
            "IMD2": (2 * Pin - self.IIP2) + self.gain,
            "IMD3": (3 * Pin - (2 * self.IIP3)) + self.gain,
            "IS": Pin - self.IS,  # TODO: get the right formula
        }


class LPF:
    def __init__(self, gain=-1, NF=1, Nthin=False):
        self.gain = gain
        self.NF = NF
        self.Nthin = Nthin

    def __call__(self, Pin, Nin):
        return {
            "Pout": Pin + self.gain,
            "Nout": Nin + (self.NF if self.Nthin else 0) + self.gain,
            "IMD2": None,
            "IMD3": None,
        }


class RFLineUp:
    def __init__(self, duplexer, lna, bpf, vga, qdemod, lpf):
        self.duplexer = duplexer
        self.lna = lna
        self.bpf = bpf
        self.vga = vga
        self.qdemod = qdemod
        self.lpf = lpf

    def __call__(self, Pin, Nin):
        # duplexer
        out = self.duplexer(Pin, Nin)
        final = {k: v for k, v in out.items() if v is not None}
        # LNA
        out = self.lna(out["Pout"], out["Nout"])
        final.update({k: v for k, v in out.items() if v is not None})
        # BPF
        out = self.bpf(out["Pout"], out["Nout"])
        final.update({k: v for k, v in out.items() if v is not None})
        # VGA
        out = self.vga(out["Pout"], out["Nout"])
        final.update({k: v for k, v in out.items() if v is not None})
        # QDEMOD
        out = self.qdemod(out["Pout"], out["Nout"])
        final.update({k: v for k, v in out.items() if v is not None})
        # LPF
        out = self.lpf(out["Pout"], out["Nout"])
        final.update({k: v for k, v in out.items() if v is not None})
        return final


def calculate_noise_figure_friis(rf_line_up):
    # block 1: duplexer
    f1 = 10 ** (rf_line_up.duplexer.NF / 10)
    g1 = 10 ** (rf_line_up.duplexer.gain / 10)
    # block 2: LNA
    f2 = 10 ** (rf_line_up.lna.NF / 10)
    g2 = 10 ** (rf_line_up.lna.gain / 10)
    # block 3: BPF
    f3 = 10 ** (rf_line_up.bpf.NF / 10)
    g3 = 10 ** (rf_line_up.bpf.gain / 10)
    # block 4: VGA
    f4 = 10 ** (rf_line_up.vga.NF / 10)
    g4 = 10 ** (rf_line_up.vga.gain / 10)
    # block 5: QDEMOD
    f5 = 10 ** (rf_line_up.qdemod.NF / 10)
    g5 = 10 ** (rf_line_up.qdemod.gain / 10)
    # block 6: LPF
    f6 = 10 ** (rf_line_up.lpf.NF / 10)
    g6 = 10 ** (rf_line_up.lpf.gain / 10)

    return f1 + (f2 - 1) / g1 + (f3 - 1) / (g1 * g2) + (f4 - 1) / (g1 * g2 * g3) + (f5 - 1) / (g1 * g2 * g3 * g4) + (f6 - 1) / (g1 * g2 * g3 * g4 * g5)


def main():
    ################### PARAMS ###################
    M = 4  # number of symbols (QPSK)
    Smin = -79  # dBm - minimum signal power
    Smax = -30  # dBm - maximum signal power
    delta = 5  # dB - margin
    Sd = Smin + delta  # dBm - desired signal power
    Iin = Smin + 13  # dBm - adjacent channel interference
    Iout = Smin + 29  # dBm - non adjacent channel interference
    bit_per_symbol = np.log2(M)  # bits per symbol
    BW = 20e6  # Hz - channel bandwidth
    Eb_No = 4.07  # dB - energy per bit to noise power spectral density ratio for 10% BER
    # dB - carrier to noise ratio
    CNRmin = Eb_No + 10 * np.log10(bit_per_symbol)

    # step 1: let's calculate the maximum noise figure of the line up
    Nfloor = Smin - CNRmin  # dBm - noise floor
    NFmax = Nfloor - (-174 + 10 * np.log10(BW))  # dB - noise figure
    print(f"Maximum noise figure of the line up: {NFmax:.2f} dB")
    # RF line up simulation
    # -> duplexer -> LNA -> RF BPF -> VGA -> QDEMOD -> LPF (for I and Q)

    # duplexer params: gain = -1 dB, NF = 1 dB
    duplexer = Duplexer(gain=-1, NF=1, Nthin=True)
    # LNA params: gain = 10 dB, NF = 3 dB, IIP2 = 0 dBm, IIP3 = 29 dBm, P1dB = 10 dBm
    lna = LNA(gain=10, NF=3, IIP2=0, IIP3=29, P1dB=10)
    # RF BPF params: gain = -3 dB, NF = 3 dB
    bpf = BPF(gain=-3, NF=3)
    # VGA params: gain = 20 dB, NF = 3 dB, IIP2 = 0 dBm, IIP3 = 29 dBm, P1dB = 10 dBm
    vga = VGA(gain=20, NF=3, IIP2=0, IIP3=29, P1dB=10)
    # QDEMOD params: gain = 7 dB, NF = 11 dB, IIP2 = 0 dBm, IIP3 = 29 dBm, IS = 40 dB
    qdemod = QDEMOD(gain=10, NF=11, IIP2=0, IIP3=29, IS=40)
    # LPF params: gain = -3 dB, NF = 3 dB
    lpf = LPF(gain=-3, NF=3)

    rf_line_up = RFLineUp(duplexer, lna, bpf, vga, qdemod, lpf)
    Nth = -174 + 10 * np.log10(BW)
    out = rf_line_up(Sd, Nth)
    G = rf_line_up.duplexer.gain + rf_line_up.lna.gain + \
        rf_line_up.bpf.gain + rf_line_up.vga.gain + \
        rf_line_up.qdemod.gain + rf_line_up.lpf.gain

    print(out)
    # Using Friis formula
    NF_equiv = 10 * np.log10(calculate_noise_figure_friis(rf_line_up))
    print(
        f"Calculated equivalent noise figure of the line up: {NF_equiv:.2f} dB")
    print(
        f"Equivalent noise figure of the line up: {(Sd - Nth) - (out['Pout'] - out['Nout'])}dB")
    print(f"Maximum NF of the line up: {NFmax:.2f}dB")
    print(f"Noise at the output of the receiver: {Nth + NF_equiv + G:.2f} dBm")

    # degradation allowance
    Dmax = Sd - CNRmin
    Da = 10**(Dmax/10) - 10**(Nfloor/10)
    Da_dBm = 10 * np.log10(Da)
    # Degradation budget
    P_phase_noise = .125 * Da
    P_spur = .125 * Da
    P_im2 = .5 * Da
    IMD2_max = 10 * np.log10(P_im2)
    P_im3 = .25 * Da
    IMD3_max = 10 * np.log10(P_im3)

    print(f"Degradation allowance:")
    print(f"  - Dmax: {Dmax:.2f} dBm")
    print(f"  - Da_dBm: {Da_dBm:.2f} dBm")
    print(f"  - N_ph_max {10 * np.log10(P_phase_noise):.2f} dBm")
    print(f"  - N_spur_max {10 * np.log10(P_spur):.2f} dBm")
    print(f"  - IMD2_max {IMD2_max:.2f} dBm")
    print(f"  - IMD3_max {IMD3_max:.2f} dBm")

    # calculating the minimum IIP2 and IIP3 (conditions: only the intermodulation products of the QMOD are considered)
    G_b4_QMOD = G - rf_line_up.qdemod.gain - rf_line_up.lpf.gain
    Iin = Iin + G_b4_QMOD
    IIP2_min = 2 * Iin - IMD2_max
    IIP3_min = (3 * Iin - IMD3_max) / 2

    print(f"Minimum IIP2 at the QDEM: {IIP2_min:.2f} dBm")
    print(f"Minimum IIP3 at the QDEM: {IIP3_min:.2f} dBm")

    # calculating the average phase noise and spurious power (dBc/Hz)


if __name__ == "__main__":
    main()
