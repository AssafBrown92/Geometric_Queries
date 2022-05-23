from scipy.special import gamma, betainc
import numpy as np

def ball_volume(r, n):
    '''
    :brief Calculates the volume of an n-dimensional ball with radius r.
    :param r The radius.
    :param n The dimension.
    :return The volume of an n-dimensional ball with radius r.
    '''
    return (r ** n) * ((np.pi ** (n/2.0)) / (gamma(1 + (n/2.0))))

def cap_volume(r,a,n):
    '''
    :brief Calculates the volume of an n-dimensional cap with radius r and arc parameter a.
    :param r The radius.
    :param a The arc parameter.
    :param n The dimension.
    :return The volume of the cap.
    '''
    ball_vol = ball_volume(r,n)
    if a >= 0:
        return 0.5 * ball_vol * betainc((n+1)/2.0, 0.5, 1-((a*a)/(r*r)))
    else:
        return ball_vol - cap_volume(r, -a, n)

def intersection_volume(c1, c2, r1, r2):
    '''
    :brief Calculates the intersection volume of the two given n-dimensional balls.
    :param c1 The coordinates of the first ball's center.
    :param c2 The coordinates of the second ball's center.
    :param r1 The radius of the first ball.
    :param r2 The radius of the second ball.
    :return The intersection volume of the two given n-dimensional balls.
    '''

    if c1.shape != c2.shape:
        raise Exception("Mismatching dimensions for center vectors")

    n = c1.shape[0]
    d = np.linalg.norm(c1 - c2)
    v1 = ball_volume(r1, n)
    v2 = ball_volume(r2, n)

    #Is there no intersection?
    if d >= r1 + r2:
        return 0

    #Is one ball subsumed in another?
    if d <= np.abs(r1 - r2):
        return min(v1, v2)

    #The balls intersect -- calculate the volume of each cap
    c1 = (d*d + r1*r1 - r2*r2) / (2*d)
    c2 = (d*d - r1*r1 + r2*r2) / (2*d)
    return cap_volume(r1, c1, n) + cap_volume(r2, c2, n)
