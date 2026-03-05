from langchain.tools import tool

@tool
def lookup_region(region_name: str) -> str:
    """
    Returns a JSON list of ISO country codes for a region like Asia or Europe.
    """
    REGION_MAP = {
        "asia": ["CN", "JP", "KR", "IN"],
        "europe": ["FR", "DE", "IT", "ES"],
        "north america": ["US", "CA", "MX"],
    }

    result = REGION_MAP.get(region_name.lower(), [])
    return str(result)