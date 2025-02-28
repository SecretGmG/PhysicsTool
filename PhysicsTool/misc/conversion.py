import math
from datetime import datetime, timedelta
from typing import Tuple

def from_MJD_to_date(MJD_list: list) -> list:
    """Converts the time from MJD format to UTC time.

    Args:
        MJD_list (list): List of dates in MJD format.

    Returns:
        list: List of converted dates in UTC time.
    """
    BASE_JD = 1721424.5  # Julian date of 1 January, 1
    MJD_TO_JD_OFFSET = 2400000.5
    dates = []
    
    for mjd in MJD_list:
        jd = mjd + MJD_TO_JD_OFFSET  # Convert MJD to JD
        days = math.floor(jd - BASE_JD)  # Calculate number of whole days
        fractional_day = jd - BASE_JD - days  # Get fractional part of the day
        date = datetime.fromordinal(days) + timedelta(days=fractional_day)  # Add fractional day
        dates.append(date)
    
    return dates

def gon_to_degrees(gons: int, tenthgons: int, centigons: int)->Tuple[int, int, int]:
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


def degrees_to_gon(degrees: int, arcminutes: int, arcseconds: int) -> Tuple[int, int, int]:
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

def display_MJD_dates(MJD_list: list) -> None:
    """Display the converted MJD dates in a nicer format (YYYY-MM-DD)."""
    dates = from_MJD_to_date(MJD_list)
    for date in dates:
        print(date.strftime("%Y-%m-%d"))
