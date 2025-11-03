"""
 * <pre>
 * The names of the facelet positions of the cube
 *             |************|
 *             |*U1**U2**U3*|
 *             |************|
 *             |*U4**U5**U6*|
 *             |************|
 *             |*U7**U8**U9*|
 *             |************|
 * ************|************|************|************|
 * *L1**L2**L3*|*F1**F2**F3*|*R1**R2**F3*|*B1**B2**B3*|
 * ************|************|************|************|
 * *L4**L5**L6*|*F4**F5**F6*|*R4**R5**R6*|*B4**B5**B6*|
 * ************|************|************|************|
 * *L7**L8**L9*|*F7**F8**F9*|*R7**R8**R9*|*B7**B8**B9*|
 * ************|************|************|************|
 *             |************|
 *             |*D1**D2**D3*|
 *             |************|
 *             |*D4**D5**D6*|
 *             |************|
 *             |*D7**D8**D9*|
 *             |************|
 * </pre>
 *
 *A cube definition string "UBL..." means for example: In position U1 we have the U-color, in position U2 we have the
 * B-color, in position U3 we have the L color etc. according to the order U1, U2, U3, U4, U5, U6, U7, U8, U9, R1, R2,
 * R3, R4, R5, R6, R7, R8, R9, F1, F2, F3, F4, F5, F6, F7, F8, F9, D1, D2, D3, D4, D5, D6, D7, D8, D9, L1, L2, L3, L4,
 * L5, L6, L7, L8, L9, B1, B2, B3, B4, B5, B6, B7, B8, B9 of the enum constants.
"""

U1 = 0
U2 = 1
U3 = 2
U4 = 3
U5 = 4
U6 = 5
U7 = 6
U8 = 7
U9 = 8
R1 = 9
R2 = 10
R3 = 11
R4 = 12
R5 = 13
R6 = 14
R7 = 15
R8 = 16
R9 = 17
F1 = 18
F2 = 19
F3 = 20
F4 = 21
F5 = 22
F6 = 23
F7 = 24
F8 = 25
F9 = 26
D1 = 27
D2 = 28
D3 = 29
D4 = 30
D5 = 31
D6 = 32
D7 = 33
D8 = 34
D9 = 35
L1 = 36
L2 = 37
L3 = 38
L4 = 39
L5 = 40
L6 = 41
L7 = 42
L8 = 43
L9 = 44
B1 = 45
B2 = 46
B3 = 47
B4 = 48
B5 = 49
B6 = 50
B7 = 51
B8 = 52
B9 = 53

facelet_values = (
    U1,
    U2,
    U3,
    U4,
    U5,
    U6,
    U7,
    U8,
    U9,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    D1,
    D2,
    D3,
    D4,
    D5,
    D6,
    D7,
    D8,
    D9,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    L9,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
    B8,
    B9,
)

facelets = {
    'U1': U1,
    'U2': U2,
    'U3': U3,
    'U4': U4,
    'U5': U5,
    'U6': U6,
    'U7': U7,
    'U8': U8,
    'U9': U9,
    'R1': R1,
    'R2': R2,
    'R3': R3,
    'R4': R4,
    'R5': R5,
    'R6': R6,
    'R7': R7,
    'R8': R8,
    'R9': R9,
    'F1': F1,
    'F2': F2,
    'F3': F3,
    'F4': F4,
    'F5': F5,
    'F6': F6,
    'F7': F7,
    'F8': F8,
    'F9': F9,
    'D1': D1,
    'D2': D2,
    'D3': D3,
    'D4': D4,
    'D5': D5,
    'D6': D6,
    'D7': D7,
    'D8': D8,
    'D9': D9,
    'L1': L1,
    'L2': L2,
    'L3': L3,
    'L4': L4,
    'L5': L5,
    'L6': L6,
    'L7': L7,
    'L8': L8,
    'L9': L9,
    'B1': B1,
    'B2': B2,
    'B3': B3,
    'B4': B4,
    'B5': B5,
    'B6': B6,
    'B7': B7,
    'B8': B8,
    'B9': B9,
}
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# The names of the edge positions of the cube. Edge UR e.g., has an U(p) and R(ight) facelet.

UR = 0
UF = 1
UL = 2
UB = 3
DR = 4
DF = 5
DL = 6
DB = 7
FR = 8
FL = 9
BL = 10
BR = 11

edge_values = (
    UR,
    UF,
    UL,
    UB,
    DR,
    DF,
    DL,
    DB,
    FR,
    FL,
    BL,
    BR,
)

edge_keys = (
    'UR',
    'UF',
    'UL',
    'UB',
    'DR',
    'DF',
    'DL',
    'DB',
    'FR',
    'FL',
    'BL',
    'BR',
)
# ++++++++++++++++++++++++++++++ Names the colors of the cube facelets ++++++++++++++++++++++++++++++++++++++++++++++++

U = 0
R = 1
F = 2
D = 3
L = 4
B = 5

color_values = (
    U,
    R,
    F,
    D,
    L,
    B,
)

color_keys = (
    'U',
    'R',
    'F',
    'D',
    'L',
    'B',
)

colors = {
    'U': U,
    'R': R,
    'F': F,
    'D': D,
    'L': L,
    'B': B,
}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# The names of the corner positions of the cube. Corner URF e.g., has an U(p), a R(ight) and a F(ront) facelet

URF = 0
UFL = 1
ULB = 2
UBR = 3
DFR = 4
DLF = 5
DBL = 6
DRB = 7

corner_values = (
    URF,
    UFL,
    ULB,
    UBR,
    DFR,
    DLF,
    DBL,
    DRB,
)

corner_keys = (
    'URF',
    'UFL',
    'ULB',
    'UBR',
    'DFR',
    'DLF',
    'DBL',
    'DRB',
)


class FaceCube(object):
    """Cube on the facelet level"""

    def __init__(self, cubeString="UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"):
        self.f = []
        for c in cubeString:
            assert (c in colors)
            self.f.append(colors[c])

    # Map the corner positions to facelet positions. cornerFacelet[URF.ordinal()][0] e.g. gives the position of the
    # facelet in the URF corner position, which defines the orientation.<br>
    # cornerFacelet[URF.ordinal()][1] and cornerFacelet[URF.ordinal()][2] give the position of the other two facelets
    # of the URF corner (clockwise).
    cornerFacelet = [
        [ U9, R1, F3 ], [ U7, F1, L3 ], [ U1, L1, B3 ], [ U3, B1, R3 ],
        [ D3, F9, R7 ], [ D1, L9, F7 ], [ D7, B9, L7 ], [ D9, R9, B7 ],
    ]

    # Map the edge positions to facelet positions. edgeFacelet[UR.ordinal()][0] e.g. gives the position of the facelet in
    # the UR edge position, which defines the orientation.<br>
    # edgeFacelet[UR.ordinal()][1] gives the position of the other facelet
    edgeFacelet = [
        [ U6, R2 ], [ U8, F2 ], [ U4, L2 ], [ U2, B2 ], [ D6, R8 ], [ D2, F8 ],
        [ D4, L8 ], [ D8, B8 ], [ F6, R4 ], [ F4, L6 ], [ B6, L4 ], [ B4, R6 ],
    ]

    # Map the corner positions to facelet colors.
    cornerColor = [
        [ U, R, F ], [ U, F, L ], [ U, L, B ], [ U, B, R ],
        [ D, F, R ], [ D, L, F ], [ D, B, L ], [ D, R, B ],
    ]

    # Map the edge positions to facelet colors.
    edgeColor = [
        [ U, R ], [ U, F ], [ U, L ],
        [ U, B ], [ D, R ], [ D, F ],
        [ D, L ], [ D, B ], [ F, R ],
        [ F, L ], [ B, L ], [ B, R ],
    ]


    # Gives string representation of a facelet cube
    def to_String(self):
        return ''.join(color_keys[c] for c in self.f)

    # Gives CubieCube representation of a faceletcube
    def toCubieCube(self):
        
        ccRet = CubieCube()
        for i in range(8):
            ccRet.cp[i] = URF   # invalidate corners
        for i in range(12):
            ccRet.ep[i] = UR    # and edges

        for i in corner_values:
            # get the colors of the cubie at corner i, starting with U/D
            for ori in range(3):
                if (self.f[self.cornerFacelet[i][ori]] == U
                        or self.f[self.cornerFacelet[i][ori]] == D):
                    break
            col1 = self.f[self.cornerFacelet[i][(ori + 1) % 3]]
            col2 = self.f[self.cornerFacelet[i][(ori + 2) % 3]]

            for j in corner_values:
                if (col1 == self.cornerColor[j][1]
                        and col2 == self.cornerColor[j][2]):
                    # in cornerposition i we have cornercubie j
                    ccRet.cp[i] = j
                    ccRet.co[i] = ori % 3
                    break

        for i in edge_values:
            for j in edge_values:
                if (self.f[self.edgeFacelet[i][0]] == self.edgeColor[j][0]
                        and self.f[self.edgeFacelet[i][1]] == self.edgeColor[j][1]):
                    ccRet.ep[i] = j
                    ccRet.eo[i] = 0
                    break

                if (self.f[self.edgeFacelet[i][0]] == self.edgeColor[j][1]
                        and self.f[self.edgeFacelet[i][1]] == self.edgeColor[j][0]):
                    ccRet.ep[i] = j
                    ccRet.eo[i] = 1
                    break

        return ccRet

import copy
from builtins import range

# n choose k
def Cnk(n, k):
    if n < k:
        return 0
    if k > n // 2:
        k = n - k
    s = 1
    i = n
    j = 1
    while i != n - k:
        s *= i
        s //= j
        i -= 1
        j += 1
    return s


def rotateLeft(arr, l, r):
    """Left rotation of all array elements between l and r"""
    temp = arr[l]
    for i in range(l, r):
        arr[i] = arr[i + 1]
    arr[r] = temp


def rotateRight(arr, l, r):
    """Right rotation of all array elements between l and r"""
    temp = arr[r]
    for i in range(r, l, -1):
        arr[i] = arr[i - 1]
    arr[l] = temp


def getURtoDF(idx1, idx2):
    """Permutation of the six edges UR,UF,UL,UB,DR,DF"""
    a = CubieCube()
    b = CubieCube()
    a.setURtoUL(idx1)
    b.setUBtoDF(idx2)
    for i in range(8):
        if a.ep[i] != BR:
            if b.ep[i] != BR:   # collision
                return -1
            else:
                b.ep[i] = a.ep[i]
    return b.getURtoDF()


class CubieCube(object):
    """Cube on the cubie level"""

    # initialize to Id-Cube

    def __init__(self, cp=None, co=None, ep=None, eo=None):
        # corner permutation
        self.cp = copy.copy(cp) if cp else [URF, UFL, ULB, UBR, DFR, DLF, DBL, DRB]

        # corner orientation
        self.co = copy.copy(co) if co else [0, 0, 0, 0, 0, 0, 0, 0]

        # edge permutation
        self.ep = copy.copy(ep) if ep else [UR, UF, UL, UB, DR, DF, DL, DB, FR, FL, BL, BR]

        # edge orientation
        self.eo = copy.copy(eo) if eo else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def toFaceCube(self):
        """return cube in facelet representation"""

        fcRet = FaceCube()
        for i in corner_values:
            j = self.cp[i]  # cornercubie with index j is at cornerposition with index i
            ori = self.co[i]   # Orientation of this cubie
            for n in range(3):
                _butya = FaceCube.cornerFacelet[i][(n + ori) % 3]
                fcRet.f[_butya] = FaceCube.cornerColor[j][n]
        for i in edge_values:
            ori = self.eo[i]   # Orientation of this cubie
            for n in range(2):
                _butya = FaceCube.edgeFacelet[i][(n + ori) % 2]
                fcRet.f[_butya] = FaceCube.edgeColor[self.ep[i]][n]
        return fcRet

    def cornerMultiply(self, b):
        """
        Multiply this CubieCube with another cubiecube b, restricted to the corners.<br>
        Because we also describe reflections of the whole cube by permutations, we get a complication with the corners. The
        orientations of mirrored corners are described by the numbers 3, 4 and 5. The composition of the orientations
        cannot
        be computed by addition modulo three in the cyclic group C3 any more. Instead the rules below give an addition in
        the dihedral group D3 with 6 elements.<br>

        NOTE: Because we do not use symmetry reductions and hence no mirrored cubes in this simple implementation of the
        Two-Phase-Algorithm, some code is not necessary here.

        b - CubieCube instance
        """

        cPerm = []  # new Corner[8]
        cOri = []   # new byte[8]
        for i in corner_values:
            cPerm.append(self.cp[b.cp[i]])

            oriA = self.co[b.cp[i]]
            oriB = b.co[i]
            ori = 0

            if oriA < 3 and oriB < 3:  # if both cubes are regular cubes...
                ori = (oriA + oriB) & 0xff   # just do an addition modulo 3 here
                if ori >= 3:
                    ori -= 3    # the composition is a regular cube

            # +++++++++++++++++++++not used in this implementation +++++++++++++
            elif oriA < 3 and oriB >= 3:    # if cube b is in a mirrored
                # state...
                ori = (oriA + oriB) & 0xff
                if ori >= 6:
                    ori -= 3    # the composition is a mirrored cube
            elif oriA >= 3 and oriB < 3:    # if cube a is an a mirrored
                # state...
                ori = (oriA - oriB) & 0xff
                if ori < 3:
                    ori += 3    # the composition is a mirrored cube
            elif oriA >= 3 and oriB >= 3:   # if both cubes are in mirrored
                # states...
                ori = (oriA - oriB) & 0xff
                if ori < 0:
                    ori += 3    # the composition is a regular cube
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            cOri.append(ori)

        for i in corner_values:
            self.cp[i] = cPerm[i]
            self.co[i] = cOri[i]

    def edgeMultiply(self, b):
        """
        Multiply this CubieCube with another cubiecube b, restricted to the edges.

        b - CubieCube instance
        """

        ePerm = []  # new Edge[12]
        eOri = []   # new byte[12]
        for i in edge_values:
            _ = b.ep[i]
            ePerm.append(self.ep[_])
            eOri.append(((b.eo[i] + self.eo[_]) % 2) & 0xff)

        for i in edge_values:
            self.ep[i] = ePerm[i]
            self.eo[i] = eOri[i]

    def multiply(self, b):
        """
        Multiply this CubieCube with another CubieCube b.

        b - CubieCube instance
        """

        self.cornerMultiply(b)
        self.edgeMultiply(b)

    def invCubieCube(self, c):
        """
        Compute the inverse CubieCube

        c - CubieCube instance
        """

        for i in edge_values:
            c.ep[self.ep[i]] = i
        for i in edge_values:
            c.eo[i] = self.eo[c.ep[i]]
        for i in corner_values:
            c.cp[self.cp[i]] = i
        for i in corner_values:
            ori = self.co[c.cp[i]]
            if ori >= 3:
                # Just for completeness. We do not invert mirrored cubes in the program.
                c.co[i] = ori
            else:
                # the standard case
                c.co[i] = -ori
                if c.co[i] < 0:
                    c.co[i] += 3

    # ********************* Get and set coordinates ***************************

    def getTwist(self):
        """return the twist of the 8 corners. 0 <= twist < 3^7"""

        ret = 0
        for i in range(URF, DRB):
            ret = (3 * ret + self.co[i]) & 0xffff
        return ret

    def setTwist(self, twist):
        twistParity = 0
        for i in range(DRB - 1, URF - 1, -1):
            self.co[i] = (twist % 3) & 0xff
            twistParity += self.co[i]
            twist //= 3
        self.co[DRB] = ((3 - twistParity % 3) % 3) & 0xff

    def getFlip(self):
        """return the flip of the 12 edges. 0<= flip < 2^11"""

        ret = 0
        for i in range(UR, BR):
            ret = (2 * ret + self.eo[i]) & 0xffff
        return ret

    def setFlip(self, flip):
        flipParity = 0
        for i in range(BR - 1, UR - 1, -1):
            self.eo[i] = (flip % 2) & 0xff
            flipParity += self.eo[i]
            flip //= 2
        self.eo[BR] = ((2 - flipParity % 2) % 2) & 0xff

    def cornerParity(self):
        """Parity of the corner permutation"""
        s = 0
        for i in range(DRB, URF, -1):
            for j in range(i - 1, URF - 1, -1):
                if self.cp[j] > self.cp[i]:
                    s += 1
        return (s % 2) & 0xffff

    def edgeParity(self):
        """Parity of the edges permutation. Parity of corners and edges are the same if the cube is solvable."""
        s = 0
        for i in range(BR, UR, -1):
            for j in range(i - 1, UR - 1, -1):
                if self.ep[j] > self.ep[i]:
                    s += 1
        return (s % 2) & 0xffff

    def getFRtoBR(self):
        """permutation of the UD-slice edges FR,FL,BL and BR"""
        a = 0
        x = 0
        edge4 = [None] * 4  # new Edge[4]
        # compute the index a < (12 choose 4) and the permutation array perm.
        for j in range(BR, UR - 1, -1):
            if (FR <= self.ep[j] and self.ep[j] <= BR):
                a += Cnk(11 - j, x + 1)
                edge4[3 - x] = self.ep[j]
                x += 1

        b = 0
        for j in range(3, 0, -1):  # compute the index b < 4! for the permutation in perm
            k = 0
            while (edge4[j] != j + 8):
                rotateLeft(edge4, 0, j)
                k += 1
            b = (j + 1) * b + k
        return (24 * a + b) & 0xffff

    def setFRtoBR(self, idx):
        sliceEdge = [FR, FL, BL, BR]
        otherEdge = [UR, UF, UL, UB, DR, DF, DL, DB]
        b = idx % 24   # Permutation
        a = idx // 24   # Combination
        for i in edge_values:
            self.ep[i] = DB     # Use UR to invalidate all edges

        for j in range(1, 4):  # generate permutation from index b
            k = b % (j + 1)
            b //= j + 1

            while k > 0:    # while (k-- > 0) #????????????????
                k -= 1
                rotateRight(sliceEdge, 0, j)

        x = 3   # generate combination and set slice edges
        for j in range(UR, BR + 1):
            if a - Cnk(11 - j, x + 1) >= 0:
                self.ep[j] = sliceEdge[3 - x]
                a -= Cnk(11 - j, x + 1)
                x -= 1
        x = 0   # set the remaining edges UR..DB
        for j in range(UR, BR + 1):
            if self.ep[j] == DB:
                self.ep[j] = otherEdge[x]
                x += 1

    def getURFtoDLF(self):
        """Permutation of all corners except DBL and DRB"""
        a = 0
        x = 0
        corner6 = []    # new Corner[6]
        # compute the index a < (8 choose 6) and the corner permutation.
        for j in range(URF, DRB + 1):
            if self.cp[j] <= DLF:
                a += Cnk(j, x + 1)
                corner6.append(self.cp[j])
                x += 1

        b = 0
        for j in range(5, 0, -1):   # compute the index b < 6! for the
            # permutation in corner6
            k = 0
            while corner6[j] != j:
                rotateLeft(corner6, 0, j)
                k += 1
            b = (j + 1) * b + k
        return (720 * a + b) & 0xffff

    def setURFtoDLF(self, idx):
        corner6 = [URF, UFL, ULB, UBR, DFR, DLF]
        otherCorner = [DBL, DRB]
        b = idx % 720  # Permutation
        a = idx // 720  # Combination
        for i in corner_values:
            self.cp[i] = DRB    # Use DRB to invalidate all corners

        for j in range(1, 6):   # generate permutation from index b
            k = b % (j + 1)
            b //= j + 1
            while k > 0:
                k -= 1
                rotateRight(corner6, 0, j)
        x = 5
        # generate combination and set corners
        for j in range(DRB, -1, -1):
            if a - Cnk(j, x + 1) >= 0:
                self.cp[j] = corner6[x]
                a -= Cnk(j, x + 1)
                x -= 1
        x = 0
        for j in range(URF, DRB + 1):
            if self.cp[j] == DRB:
                self.cp[j] = otherCorner[x]
                x += 1

    def getURtoDF(self):
        """Permutation of the six edges UR,UF,UL,UB,DR,DF."""
        a = 0
        x = 0
        edge6 = []  # new Edge[6]
        # compute the index a < (12 choose 6) and the edge permutation.
        for j in range(UR, BR + 1):
            if self.ep[j] <= DF:
                a += Cnk(j, x + 1)
                edge6.append(self.ep[j])
                x += 1

        b = 0
        for j in range(5, 0, -1):  # compute the index b < 6! for the permutation in edge6
            k = 0
            while edge6[j] != j:
                rotateLeft(edge6, 0, j)
                k += 1
            b = (j + 1) * b + k
        return 720 * a + b

    def setURtoDF(self, idx):
        edge6 = [UR, UF, UL, UB, DR, DF]
        otherEdge = [DL, DB, FR, FL, BL, BR]
        b = idx % 720  # Permutation
        a = idx // 720  # Combination
        for i in edge_values:
            self.ep[i] = BR     # Use BR to invalidate all edges

        for j in range(1, 6):   # generate permutation from index b
            k = b % (j + 1)
            b //= j + 1
            while k > 0:
                k -= 1
                rotateRight(edge6, 0, j)
        x = 5
        # generate combination and set edges
        for j in range(BR, -1, -1):
            if a - Cnk(j, x + 1) >= 0:
                self.ep[j] = edge6[x]
                a -= Cnk(j, x + 1)
                x -= 1
        x = 0
        # set the remaining edges DL..BR
        for j in range(UR, BR + 1):
            if self.ep[j] == BR:
                self.ep[j] = otherEdge[x]
                x += 1

    def getURtoUL(self):
        """Permutation of the three edges UR,UF,UL"""
        a = 0
        x = 0
        edge3 = []  # new Edge[3]
        # compute the index a < (12 choose 3) and the edge permutation.
        for j in range(UR, BR + 1):
            if self.ep[j] <= UL:
                a += Cnk(j, x + 1)
                edge3.append(self.ep[j])
                x += 1

        b = 0
        for j in range(2, 0, -1):  # compute the index b < 3! for the permutation in edge3
            k = 0
            while edge3[j] != j:
                rotateLeft(edge3, 0, j)
                k += 1
            b = (j + 1) * b + k
        return (6 * a + b) & 0xffff

    def setURtoUL(self, idx):
        edge3 = [UR, UF, UL]
        b = idx % 6    # Permutation
        a = idx // 6    # Combination
        for i in edge_values:
            self.ep[i] = BR    # Use BR to invalidate all edges

        for j in range(1, 3):   # generate permutation from index b
            k = b % (j + 1)
            b //= j + 1
            while k > 0:
                k -= 1
                rotateRight(edge3, 0, j)
        x = 2  # generate combination and set edges
        for j in range(BR, -1, -1):
            if a - Cnk(j, x + 1) >= 0:
                self.ep[j] = edge3[x]
                a -= Cnk(j, x + 1)
                x -= 1

    def getUBtoDF(self):
        """Permutation of the three edges UB,DR,DF"""
        a = 0
        x = 0
        edge3 = []  # new Edge[3]
        # compute the index a < (12 choose 3) and the edge permutation.
        for j in range(UR, BR + 1):
            if UB <= self.ep[j] and self.ep[j] <= DF:
                a += Cnk(j, x + 1)
                edge3.append(self.ep[j])
                x += 1

        b = 0
        for j in range(2, 0, -1):  # compute the index b < 3! for the permutation in edge3
            k = 0
            while edge3[j] != UB + j:
                rotateLeft(edge3, 0, j)
                k += 1
            b = (j + 1) * b + k
        return (6 * a + b) & 0xffff

    def setUBtoDF(self, idx):
        edge3 = [UB, DR, DF]
        b = idx % 6    # Permutation
        a = idx // 6    # Combination
        for i in edge_values:
            self.ep[i] = BR     # Use BR to invalidate all edges

        for j in range(1, 3):
            # generate permutation from index b
            k = b % (j + 1)
            b //= j + 1
            while k > 0:
                k -= 1
                rotateRight(edge3, 0, j)
        x = 2
        # generate combination and set edges
        for j in range(BR, -1, -1):
            if a - Cnk(j, x + 1) >= 0:
                self.ep[j] = edge3[x]
                a -= Cnk(j, x + 1)
                x -= 1

    def getURFtoDLB(self):
        perm = copy.copy(self.cp)
        b = 0
        for j in range(7, 0, -1):  # compute the index b < 8! for the permutation in perm
            k = 0
            while perm[j] != j:
                rotateLeft(perm, 0, j)
                k += 1
            b = (j + 1) * b + k
        return b

    def setURFtoDLB(self, idx):
        perm = [URF, UFL, ULB, UBR, DFR, DLF, DBL, DRB]
        for j in range(1, 8):
            k = idx % (j + 1)
            idx //= j + 1
            while k > 0:
                k -= 1
                rotateRight(perm, 0, j)
        x = 7
        # set corners
        for j in range(7, -1, -1):
            self.cp[j] = perm[x]
            x -= 1

    def getURtoBR(self):
        perm = copy.copy(self.ep)
        b = 0
        for j in range(11, 0, -1):     # compute the index b < 12! for the permutation in perm
            k = 0
            while perm[j] != j:
                rotateLeft(perm, 0, j)
                k += 1
            b = (j + 1) * b + k
        return b

    def setURtoBR(self, idx):
        perm = [UR, UF, UL, UB, DR, DF, DL, DB, FR, FL, BL, BR]
        for j in range(1, 12):
            k = idx % (j + 1)
            idx //= j + 1
            while k > 0:
                k -= 1
                rotateRight(perm, 0, j)
        x = 11  # set edges
        for j in range(11, -1, -1):
            self.ep[j] = perm[x]
            x -= 1

    def verify(self):
        """
        Check a cubiecube for solvability. Return the error code.
        0: Cube is solvable
        -2: Not all 12 edges exist exactly once
        -3: Flip error: One edge has to be flipped
        -4: Not all corners exist exactly once
        -5: Twist error: One corner has to be twisted
        -6: Parity error: Two corners ore two edges have to be exchanged
        """

        sum = 0
        edgeCount = [0] * 12  # new int[12]
        for i in edge_values:
            edgeCount[self.ep[i]] += 1
        for i in range(12):
            if edgeCount[i] != 1:
                return -2

        for i in range(12):
            sum += self.eo[i]
        if sum % 2 != 0:
            return -3

        cornerCount = [0] * 8   # new int[8]
        for i in corner_values:
            cornerCount[self.cp[i]] += 1
        for i in range(8):
            if cornerCount[i] != 1:
                return -4   # missing corners

        sum = 0
        for i in range(8):
            sum += self.co[i]
        if sum % 3 != 0:
            return -5   # twisted corner

        if (self.edgeParity() ^ self.cornerParity()) != 0:
            return -6   # parity error

        return 0    # cube ok


# ************************ Moves on the cubie level ****************************

cpU = [UBR, URF, UFL, ULB, DFR, DLF, DBL, DRB]
coU = [0, 0, 0, 0, 0, 0, 0, 0]
epU = [UB, UR, UF, UL, DR, DF, DL, DB, FR, FL, BL, BR]
eoU = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

cpR = [DFR, UFL, ULB, URF, DRB, DLF, DBL, UBR]
coR = [2, 0, 0, 1, 1, 0, 0, 2]
epR = [FR, UF, UL, UB, BR, DF, DL, DB, DR, FL, BL, UR]
eoR = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cpF = [UFL, DLF, ULB, UBR, URF, DFR, DBL, DRB]
coF = [1, 2, 0, 0, 2, 1, 0, 0]
epF = [UR, FL, UL, UB, DR, FR, DL, DB, UF, DF, BL, BR]
eoF = [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]
cpD = [URF, UFL, ULB, UBR, DLF, DBL, DRB, DFR]
coD = [0, 0, 0, 0, 0, 0, 0, 0]
epD = [UR, UF, UL, UB, DF, DL, DB, DR, FR, FL, BL, BR]
eoD = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cpL = [URF, ULB, DBL, UBR, DFR, UFL, DLF, DRB]
coL = [0, 1, 2, 0, 0, 2, 1, 0]
epL = [UR, UF, BL, UB, DR, DF, FL, DB, FR, UL, DL, BR]
eoL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cpB = [URF, UFL, UBR, DRB, DFR, DLF, ULB, DBL]
coB = [0, 0, 1, 2, 0, 0, 2, 1]
epB = [UR, UF, UL, BR, DR, DF, DL, BL, FR, FL, UB, DB]
eoB = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]

# this CubieCube array represents the 6 basic cube moves
moveCube = [
    CubieCube(cp=cpU, co=coU, ep=epU, eo=eoU),
    CubieCube(cp=cpR, co=coR, ep=epR, eo=eoR),
    CubieCube(cp=cpF, co=coF, ep=epF, eo=eoF),
    CubieCube(cp=cpD, co=coD, ep=epD, eo=eoD),
    CubieCube(cp=cpL, co=coL, ep=epL, eo=eoL),
    CubieCube(cp=cpB, co=coB, ep=epB, eo=eoB),
]
