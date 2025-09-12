from typing import Tuple
import numpy as np


def mjd_to_datetime(mjd: np.ndarray, start: np.datetime64 | None = None) -> np.ndarray:
    """
    Parses Modified Julian Date (MJD) to datetime64[ns] or timedelta64[ns].

    Args:
        mjd: Array of MJD values (float or int).
        start: Optional custom start date.
               If None, defaults to MJD epoch (1858-11-17).
               If 0 returns timedelta64[ns] from .

    Returns:
        Array of datetime64[ns] or timedelta64[ns] values.
    """
    mjd_ns = (np.asarray(mjd, dtype=np.float64) * 86_400_000_000_000).astype(
        "timedelta64[ns]"
    )

    if start is None:
        start = np.datetime64("1858-11-17T00:00:00", "ns")

    return start + mjd_ns


def gon_to_degrees(gons: int, tenthgons: int, centigons: int) -> Tuple[int, int, float]:
    """Convert from gons to degrees, arcminutes, and arcseconds.

    Args:
        gons (int): Number of gons.
        tenthgons (int, optional): Number of tenthgons (1/10 of a gon).
        centigons (int, optional): Number of centigons (1/100 of a gon).

    Returns:
        tuple: Degrees (°, '), arcminutes ('), arcseconds (").
    """
    # Total gons as a decimal number
    total_gons = gons + tenthgons / 10 + centigons / 100

    # Convert gons to degrees
    total_degrees = total_gons * (360 / 400)

    # Extract degrees, arcminutes, and arcseconds
    degrees = int(total_degrees)
    arcminutes = int((total_degrees - degrees) * 60)
    arcseconds = (total_degrees - degrees - arcminutes / 60) * 3600

    return (degrees, arcminutes, arcseconds)


def degrees_to_gon(
    degrees: int, arcminutes: int, arcseconds: float
) -> Tuple[int, int, int]:
    """Convert from degrees, arcminutes, and arcseconds to gons, tenthgons, and centigons.

     Args:
        degrees (int): Number of degrees.
        arcminutes (int, optional): Number of arcminutes (1/60 of a degree). Defaults to 0.
        arcseconds (float, optional): Number of arcseconds (1/3600 of a degree). Defaults to 0.

    Returns:
        tuple: Gons, tenthgons, centigons.
    """
    # Total degrees as a decimal number
    total_degrees = degrees + arcminutes / 60 + arcseconds / 3600

    # Convert degrees to gons
    total_gons = total_degrees * (400 / 360)

    # Extract gons, tenthgons, and centigons
    gons = int(total_gons)
    tenthgons = int((total_gons - gons) * 10)
    centigons = round((total_gons - gons - tenthgons / 10) * 100)

    return (gons, tenthgons, centigons)


def display_gon_to_degrees(gons: int, tenthgons: int, centigons: int) -> str:
    """Display the conversion from gons to degrees, arcminutes, and arcseconds with units."""
    degrees, arcminutes, arcseconds = gon_to_degrees(gons, tenthgons, centigons)
    return f"{gons}g {tenthgons}tg {centigons}cg = {degrees}° {arcminutes}' {arcseconds:.2f}\""


def display_degrees_to_gon(degrees: int, arcminutes: int, arcseconds: float) -> str:
    """Display the conversion from degrees, arcminutes, and arcseconds to gons, tenthgons, and centigons with units."""
    gons, tenthgons, centigons = degrees_to_gon(degrees, arcminutes, arcseconds)
    return f"{degrees}° {arcminutes}' {arcseconds:.2f}\" = {gons}g {tenthgons}tg {centigons}cg"
