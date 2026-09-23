"""Solids, tubes and STEP files from the fitted surfaces. Needs OCP.

The topology rules of ROADMAP-CAD.md section 2.2 are made here:

  * one face per smooth piece -- a wing panel is a skin, a trailing edge
    (or a hinge cut) and two end caps, four faces;
  * the skin wraps round the nose, so its seam is at the trailing edge
    and the leading edge is the smooth interior of a face;
  * end caps are PLANES and tubes CYLINDERS, analytic surfaces a person
    can select, dimension and mate to, not B-splines that happen to be
    flat or round;
  * adjacent faces are built from the same boundary poles, so they share
    their edges exactly and sew into one closed, valid solid;
  * the file is an XCAF document: every part, and every face, has a name.

## Which half a printed part is

The print frame is LEFT-handed against the flight frame: X is flight x,
Y is height square to the panel axis and Z runs outboard, and (x, height,
outboard) is a reflection of (x, y, z) for the right wing. So an STL
exactly as exported is the LEFT half of the aircraft, placed by a proper
rotation, and the right half is its mirror image -- which is why the
build sheet says to print two of each and mirror one. The assembly here
builds the right half as mirrored GEOMETRY, not a mirrored instance: a
negative-determinant placement is exactly the thing many CAD importers
get wrong.
"""

from __future__ import annotations

import numpy as np

from OCP.BRep import BRep_Builder
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge,
                                BRepBuilderAPI_MakeFace,
                                BRepBuilderAPI_MakeSolid,
                                BRepBuilderAPI_MakeWire,
                                BRepBuilderAPI_Sewing,
                                BRepBuilderAPI_Transform)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.GProp import GProp_GProps
from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.Interface import Interface_Static
from OCP.Quantity import Quantity_Color, Quantity_TypeOfColor
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_StepModelType
from OCP.ShapeFix import ShapeFix_Solid
from OCP.collections import (Array1_double, Array1_gp_Pnt, Array1_int,
                             Array2_gp_Pnt)
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

from .bspline import Surface

COLOURS = {
    "vase": (0.93, 0.93, 0.90),     # foamed PLA, printed spiral
    "solid": (0.95, 0.55, 0.15),    # printed solid companions
    "carbon": (0.12, 0.12, 0.13),   # bought tube
    "payload": (0.25, 0.55, 0.85),  # reserved volume, not a part
}


# ------------------------------------------------------------ geometry


def _knots(knots: np.ndarray):
    k, m = Surface.knots_and_mults(knots)
    ka = Array1_double(1, len(k))
    ma = Array1_int(1, len(m))
    for i, (kk, mm) in enumerate(zip(k, m), start=1):
        ka.SetValue(i, float(kk))
        ma.SetValue(i, int(mm))
    return ka, ma


def occ_surface(s: Surface) -> Geom_BSplineSurface:
    nu, nv, _ = s.poles.shape
    P = Array2_gp_Pnt(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            x, y, z = s.poles[i, j]
            P.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))
    uk, um = _knots(s.u_knots)
    vk, vm = _knots(s.v_knots)
    return Geom_BSplineSurface(P, uk, vk, um, vm, s.u_deg, s.v_deg)


def occ_curve(poles: np.ndarray, knots: np.ndarray, deg: int) -> Geom_BSplineCurve:
    P = Array1_gp_Pnt(1, len(poles))
    for i, (x, y, z) in enumerate(poles, start=1):
        P.SetValue(i, gp_Pnt(float(x), float(y), float(z)))
    k, m = _knots(knots)
    return Geom_BSplineCurve(P, k, m, deg)


def _shapes(shape, kind):
    out, ex = [], TopExp_Explorer(shape, kind)
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    return out


def part_solid(surfs: list[Surface]) -> tuple[object, dict]:
    """The closed solid of one part, from its pieces, in its own frame.

    -> (solid, {"faces": [(name, face), ...]}). Each piece is one face on
    its natural bounds; each end is one planar face bounded by the same
    boundary curves the pieces carry, so every edge is shared by exactly
    two faces once sewn."""
    named = [(s.name, BRepBuilderAPI_MakeFace(occ_surface(s), 1e-6).Face())
             for s in surfs]
    for end, label in ((0, "root face"), (-1, "tip face")):
        wire = BRepBuilderAPI_MakeWire()
        for s in surfs:
            wire.Add(BRepBuilderAPI_MakeEdge(
                occ_curve(s.poles[:, end, :], s.u_knots, s.u_deg)).Edge())
        named.append((label, BRepBuilderAPI_MakeFace(wire.Wire(), True).Face()))
    sew = BRepBuilderAPI_Sewing(1e-3)
    for _, f in named:
        sew.Add(f)
    sew.Perform()
    shells = _shapes(sew.SewedShape(), TopAbs_SHELL)
    if len(shells) != 1:
        raise RuntimeError(f"sewing gave {len(shells)} shells, not one")
    solid = BRepBuilderAPI_MakeSolid(TopoDS.Shell(shells[0])).Solid()
    fix = ShapeFix_Solid(solid)
    fix.Perform()
    solid = _shapes(fix.Shape(), TopAbs_SOLID)[0]
    # the sewn faces are new objects: name them by matching each to the
    # piece it came from by its centre of area
    faces = _shapes(solid, TopAbs_FACE)
    names = []
    ref = [(n, _centre(f)) for n, f in named]
    for f in faces:
        c = _centre(f)
        names.append(min(ref, key=lambda r: float(np.linalg.norm(r[1] - c)))[0])
    return solid, {"faces": list(zip(names, faces))}


def _centre(face) -> np.ndarray:
    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, g)
    p = g.CentreOfMass()
    return np.array([p.X(), p.Y(), p.Z()])


def volume_mm3(shape) -> float:
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return float(abs(g.Mass()))


def is_valid(shape) -> bool:
    return bool(BRepCheck_Analyzer(shape).IsValid())


def tube(p0, direction, length_mm: float, od_mm: float, id_mm: float):
    """A carbon tube: two coaxial cylinders, the inner cut from the outer,
    so its faces are CYLINDRICAL_SURFACEs and PLANEs."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    ax = gp_Ax2(gp_Pnt(*map(float, p0)), gp_Dir(*map(float, d)))
    outer = BRepPrimAPI_MakeCylinder(ax, 0.5 * od_mm, float(length_mm)).Shape()
    if id_mm <= 0.0:
        return outer
    inner = BRepPrimAPI_MakeCylinder(ax, 0.5 * id_mm, float(length_mm)).Shape()
    return BRepAlgoAPI_Cut(outer, inner).Shape()


def box(x0, x1, y0, y1, z0, z1):
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), gp_Pnt(x1, y1, z1)).Shape()


# ---------------------------------------------------------- placement


def print_to_flight_left(frame, origin_mm) -> gp_Trsf:
    """The proper rotation that puts a part, as printed, where the LEFT
    half of it flies. x = X + ox; with Y' = Y + oy and s = Z,
    y = -(y_p + s cos(phi) - Y' sin(phi)), z = z_p + s sin(phi) + Y' cos(phi)."""
    c, s = frame.cos, frame.sin
    ay, az = frame.origin_yz_mm
    ox, oy = origin_mm
    t = gp_Trsf()
    t.SetValues(1.0, 0.0, 0.0, float(ox),
                0.0, s, -c, float(s * oy - ay),
                0.0, c, s, float(az + c * oy))
    return t


def _plane_of(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(centroid, unit normal) of the plane through `points`, by SVD."""
    c = points.mean(0)
    n = np.linalg.svd(points - c)[2][-1]
    return c, n / np.linalg.norm(n)


def insert_solid(ins) -> tuple[object, dict]:
    """A joint wedge insert (printing/inserts.py) as a clean solid, in the
    flight frame (the right wing's).

    Two wires -- face A's outline and face B's, each an interpolated arc
    closed by the straight crest cut -- lofted RULED: the skin between the
    arcs is one ruled B-spline face, exactly the straight rulings the STL
    is built of, and the crest cut between the two lines is another. The
    caps are planes. Each tube's bore is a true cylinder cut through it.
    -> (solid, {"faces": [(name, face), ...]})."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.collections import HArray1_gp_Pnt

    # The outline is smooth except where it is not: a joint at the elevon's
    # root runs round the hinge cut, two sharp corners, and ONE spline
    # through them overshot and took 12% off the solid's volume. So the
    # arc is split at its corners -- the same indices on both faces, which
    # correspond point for point -- one spline per smooth run, a line for
    # a straight one. The panel CAD learned this first (bspline.py).
    a = ins.arc_a
    d1 = np.diff(a, axis=0)
    d1 /= np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-12)
    turn = np.degrees(np.arccos(np.clip((d1[:-1] * d1[1:]).sum(1), -1.0, 1.0)))
    cuts = [0] + [int(i) + 1 for i in np.flatnonzero(turn > 25.0)] + [len(a) - 1]

    def edge(run: np.ndarray):
        if len(run) <= 2:
            return BRepBuilderAPI_MakeEdge(gp_Pnt(*map(float, run[0])),
                                           gp_Pnt(*map(float, run[-1]))).Edge()
        pts = HArray1_gp_Pnt(1, len(run))
        for i, (x, y, z) in enumerate(run, start=1):
            pts.SetValue(i, gp_Pnt(float(x), float(y), float(z)))
        it = GeomAPI_Interpolate(pts, False, 1e-7)
        it.Perform()
        return BRepBuilderAPI_MakeEdge(it.Curve()).Edge()

    def wire(arc: np.ndarray):
        w = BRepBuilderAPI_MakeWire()
        for i0, i1 in zip(cuts, cuts[1:]):
            w.Add(edge(arc[i0:i1 + 1]))
        w.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(*map(float, arc[-1])),
                                      gp_Pnt(*map(float, arc[0]))).Edge())
        return w.Wire()

    loft = BRepOffsetAPI_ThruSections(True, True, 1e-6)
    loft.AddWire(wire(ins.arc_a))
    loft.AddWire(wire(ins.arc_b))
    loft.CheckCompatibility(False)
    loft.Build()
    solid = loft.Shape()
    y_j = float(ins.arc_a[:, 1].mean())
    for p, d, r in ins.bore_axes:
        at = p + d * ((y_j - p[1]) / d[1]) - 40.0 * d
        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*map(float, at)),
                                              gp_Dir(*map(float, d))), float(r), 80.0).Shape()
        solid = BRepAlgoAPI_Cut(solid, cyl).Shape()
    solids = _shapes(solid, TopAbs_SOLID)
    if len(solids) == 1:
        solid = solids[0]

    ca, na = _plane_of(ins.arc_a)
    cb, nb = _plane_of(ins.arc_b)
    faces = _shapes(solid, TopAbs_FACE)
    named, rest = [], []
    for f in faces:
        kind = BRepAdaptor_Surface(TopoDS.Face(f)).GetType()
        c = _centre(f)
        if kind == GeomAbs_Cylinder:
            named.append(("bore", f))
        elif kind == GeomAbs_Plane and abs((c - ca) @ na) < 1e-3:
            named.append(("face A (outer panel root)", f))
        elif kind == GeomAbs_Plane and abs((c - cb) @ nb) < 1e-3:
            named.append(("face B (inner panel tip)", f))
        else:
            rest.append(f)

    def area(f):
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, g)
        return g.Mass()
    # What is left is skin and two kinds of flat cut: the crest cut between
    # the ends of the two arcs, and a hinge cut wherever the outline has a
    # straight run. A ruled loft makes both as B-spline faces, not planes,
    # so they are named by WHERE they are: the face whose centre is nearest
    # a cut's own centre, within 2 mm.
    b = ins.arc_b
    targets = [("crest cut", 0.25 * (a[0] + a[-1] + b[0] + b[-1]))]
    for i0, i1 in zip(cuts, cuts[1:]):
        if i1 - i0 == 1:
            targets.append(("hinge cut", 0.25 * (a[i0] + a[i1] + b[i0] + b[i1])))
    left = list(rest)
    for label, c_t in targets:
        if not left:
            break
        k = min(range(len(left)), key=lambda i: np.linalg.norm(_centre(left[i]) - c_t))
        if np.linalg.norm(_centre(left[k]) - c_t) < 2.0:
            named.append((label, left.pop(k)))
    left.sort(key=area, reverse=True)
    named += [("skin", f) for f in left]
    return solid, {"faces": named}


def transform_rt(shape, R: np.ndarray, t: np.ndarray):
    """A proper rigid motion p -> R p + t, as new geometry."""
    tr = gp_Trsf()
    tr.SetValues(*(float(v) for v in (R[0, 0], R[0, 1], R[0, 2], t[0],
                                       R[1, 0], R[1, 1], R[1, 2], t[1],
                                       R[2, 0], R[2, 1], R[2, 2], t[2])))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def transformed(shape, trsf: gp_Trsf):
    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def mirrored_y(shape):
    """The mirror image in the symmetry plane, as new geometry."""
    t = gp_Trsf()
    t.SetMirror(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)))
    return BRepBuilderAPI_Transform(shape, t, True).Shape()


# ---------------------------------------------------------- the checks


def min_distance_mm(a, b) -> float:
    d = BRepExtrema_DistShapeShape(a, b)
    d.Perform()
    return float(d.Value())


def point_inside(solid, p) -> bool:
    cl = BRepClass3d_SolidClassifier(solid, gp_Pnt(*map(float, p)), 1e-6)
    return cl.State() == TopAbs_IN


def compound(shapes):
    b = BRep_Builder()
    c = TopoDS_Compound()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


# ---------------------------------------------------------- STEP output


class StepDocument:
    """An XCAF document: named, coloured parts, each face named too.

    `add(name, shape, colour, faces=[(name, face), ...])` and `write`.
    AP214 in millimetres, which every CAD package reads; the face names
    go out as sub-shape names."""

    def __init__(self, title: str):
        app = XCAFApp_Application.GetApplication_s()
        self.doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
        app.InitDocument(self.doc)
        self.shapes = XCAFDoc_DocumentTool.ShapeTool_s(self.doc.Main())
        self.colours = XCAFDoc_DocumentTool.ColorTool_s(self.doc.Main())
        self.title = title

    def add(self, name: str, shape, colour: str = "vase", faces=()):
        label = self.shapes.AddShape(shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))
        r, g, b = COLOURS.get(colour, COLOURS["vase"])
        self.colours.SetColor(label, Quantity_Color(r, g, b, Quantity_TypeOfColor.Quantity_TOC_RGB),
                              XCAFDoc_ColorType.XCAFDoc_ColorSurf)
        for fname, face in faces:
            sub = self.shapes.AddSubShape(label, face)
            if not sub.IsNull():
                TDataStd_Name.Set_s(sub, TCollection_ExtendedString(fname))
        return label

    def write(self, path) -> bool:
        # The writer FIRST: creating the first one in a process loads the
        # STEP defaults over any parameter set before it, and the face names
        # of the first file written went with them.
        w = STEPCAFControl_Writer()
        Interface_Static.SetCVal_s("write.step.schema", "AP214IS")
        Interface_Static.SetCVal_s("write.step.unit", "MM")
        Interface_Static.SetIVal_s("write.stepcaf.subshapes.name", 1)
        Interface_Static.SetCVal_s("write.step.product.name", self.title)
        w.SetNameMode(True)
        w.SetColorMode(True)
        w.Transfer(self.doc, STEPControl_StepModelType.STEPControl_AsIs)
        return w.Write(str(path)) == IFSelect_ReturnStatus.IFSelect_RetDone
