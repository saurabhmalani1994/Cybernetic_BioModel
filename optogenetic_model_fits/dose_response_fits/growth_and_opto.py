import numpy as np

def parFun_growth(alpha=1.0, beta=0.2, alpha_mult = [1., 1., 1.], beta_mult = [1., 1., 1.],
         mu1_max=0.45, mu2_max=0.2, mu3_max=0.33, Y1=0.15, Y2=0.74, Y3=0.50, K1=0.1, K2=0.02, K3=0.001,
         phi1=0.41, phi2=1.067, phi3=2.087, phi4=0.95, kLa=350.0, KO2=0.0001, KO3=0.0001, gamma1=10.0, gamma2=10.0,
         gamma3=0.1, alpha_star=[0.1,0.1,0.1], O_star=7.5):

         return alpha, beta, mu1_max, mu2_max, mu3_max, Y1, Y2, Y3, K1, K2, K3, phi1, phi2, phi3, phi4, kLa, \
                    KO2, KO3, gamma1, gamma2, gamma3, alpha_star, O_star, alpha_mult, beta_mult

def parFun_opto(TFtot = 2000, kon = 0.0016399, koff = 0.34393, kbasal = 0.02612, kmax = 13.588, Kd = 956.75,
                    n = 4.203, kdegR = 0.042116, ktrans = 1.4514, kdegP = 0.007):
    return kon, TFtot, koff, kbasal, kmax, n, Kd, kdegR, ktrans, kdegP

def odeFun(t, vars, D, I, G0 = None, pars_growth = None, pars_opto = None):
    # print(t)
    if callable(I):
        I = I(t)

    if G0 is None:
        G0 = 28
    
    X, G, E, O, e1, e2, e3, C, TFon, mRNA = vars

    if pars_growth is None:
        alpha, beta, mu1_max, mu2_max, mu3_max, Y1, Y2, Y3, K1, K2, K3, phi1, phi2, phi3, phi4, kLa, \
                        KO2, KO3, gamma1, gamma2, gamma3, alpha_star, O_star, alpha_mult, beta_mult = parFun_growth()
    else:
        alpha, beta, mu1_max, mu2_max, mu3_max, Y1, Y2, Y3, K1, K2, K3, phi1, phi2, phi3, phi4, kLa, \
                        KO2, KO3, gamma1, gamma2, gamma3, alpha_star, O_star, alpha_mult, beta_mult = pars_growth
    
    if pars_opto is None:
        kon, TFtot, koff, kbasal, kmax, n, Kd, kdegR, ktrans, kdegP = parFun_opto()
    else:
        kon, TFtot, koff, kbasal, kmax, n, Kd, kdegR, ktrans, kdegP = pars_opto

    mu1 = mu1_max * (mu1_max + beta) / (alpha + alpha_star[0])
    mu2 = mu2_max * (mu2_max + beta) / (alpha + alpha_star[1])
    mu3 = mu3_max * (mu3_max + beta) / (alpha + alpha_star[2])

    if G > 0:
        r1 = mu1 * G / (K1 + G)
        if O > 0:
            r3 = mu3 * (G / (K3 + G)) * (O / (KO3 + O))
        else:
            r3 = 0
    else:
        r1 = 0
        r3 = 0

    if E > 0:
        r2 = mu2 * (E / (K2 + E)) * (O / (KO2 + O))
    else:
        r2 = 0

    # r1 = r1 * e1
    # r2 = r2 * e2
    # r3 = r3 * e3

    u1 = r1 / ((r1 + r2 + r3) + 1e-10)
    u2 = r2 / ((r1 + r2 + r3) + 1e-10)
    u3 = r3 / ((r1 + r2 + r3) + 1e-10)

    v1 = r1 / (np.max(r1 + r2 + r3) + 1e-10)
    v2 = r2 / (np.max(r1 + r2 + r3) + 1e-10)
    v3 = r3 / (np.max(r1 + r2 + r3) + 1e-10)

    r1 = r1 * e1
    r2 = r2 * e2
    r3 = r3 * e3
    sumr1v1 = (r1*v1 + r2*v2 + r3*v3)

    # dXdt
    dXdt = ((sumr1v1) - D) * X

    dCdt = gamma3*r3*v3 - (gamma1*r1*v1 + gamma2*r2*v2) * C - sumr1v1 * C
    # dCdt = 0

    if G > 0:
        dGdt = (G0 - G) * D - (r1*v1/Y1 + r3*v3/Y3)*X - phi4*(C*dXdt + dCdt)
    else:
        dGdt = (G0 ) * D - (r1*v1/Y1 + r3*v3/Y3)*X - phi4*(C*dXdt + dCdt)

    if E > 0:
        dEdt = -D*E + (phi1*r1*v1/Y1 - r2*v2/Y2) * X
    else:
        dEdt = (phi1*r1*v1/Y1 - r2*v2/Y2) * X

    dOdt = kLa * (O_star - O) - (phi2*r2*v2/Y2 + phi3*r3*v3/Y3) * X
    # dOdt = 0

    if G > 0:
        de1dt = alpha * mRNA * alpha_mult[0] * u1 * (G / (K1 + G)) - (sumr1v1 + beta * beta_mult[0]) * e1 + alpha_star[0]
        de3dt = alpha * alpha_mult[2] * u3 * (G / (K3 + G)) - (sumr1v1 + beta * beta_mult[2]) * e3 + alpha_star[2]
    else:
        # print('G gone')
        de1dt = - (sumr1v1 + beta * beta_mult[0]) * e1 + alpha_star[0]
        de3dt = - (sumr1v1 + beta * beta_mult[2]) * e3 + alpha_star[2]

    if E > 0:
        de2dt = alpha * alpha_mult[1] * u2 * (E / (K2 + E)) - (sumr1v1 + beta * beta_mult[1]) * e2 + alpha_star[1]
    else:
        # print('E gone too')
        de2dt = - (sumr1v1 + beta * beta_mult[1]) * e2 + alpha_star[1]
    

    TFon = np.max([0,TFon])
    mRNA = mRNA * (kbasal + kmax) / kdegR

    dTFondt = I * kon * (TFtot - TFon) - koff * TFon
    dmRNAdt = kbasal + kmax * (TFon ** n) / ((Kd ** n) + (TFon ** n)) - (kdegR) * mRNA
    # changing to per hour basis
    dTFondt = dTFondt * 60
    dmRNAdt = dmRNAdt * 60

    dmRNAdt = dmRNAdt * kdegR / (kbasal + kmax)

    output = dXdt, dGdt, dEdt, dOdt, de1dt, de2dt, de3dt, dCdt, dTFondt, dmRNAdt

    return output

