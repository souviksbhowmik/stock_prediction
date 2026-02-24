"""NIFTY 50, NIFTY Next 50, and popular mid-cap tickers with alias maps."""

NIFTY_50_TICKERS: list[str] = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "LTIM.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SHREECEM.NS", "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "WIPRO.NS",
]

# NIFTY Next 50 — stocks outside NIFTY 50 but in NIFTY 100
NIFTY_NEXT_50_TICKERS: list[str] = [
    "ABCAPITAL.NS", "AMBUJACEM.NS", "AUBANK.NS", "BANKBARODA.NS",
    "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS", "CANBK.NS",
    "CHOLAFIN.NS", "COLPAL.NS", "CONCOR.NS", "CUMMINSIND.NS",
    "DLF.NS", "GAIL.NS", "GODREJCP.NS", "GODREJPROP.NS",
    "HAVELLS.NS", "HINDPETRO.NS", "ICICIGI.NS", "ICICIPRULI.NS",
    "IGL.NS", "INDUSTOWER.NS", "IOC.NS", "IRCTC.NS",
    "JINDALSTEL.NS", "JUBLFOOD.NS", "LICI.NS", "LUPIN.NS",
    "MARICO.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NAUKRI.NS",
    "NMDC.NS", "PAGEIND.NS", "PETRONET.NS", "PIDILITIND.NS",
    "PNB.NS", "SBICARD.NS", "SIEMENS.NS", "TATACOMM.NS",
    "TATAPOWER.NS", "TORNTPHARM.NS", "TRENT.NS", "VEDL.NS",
    "VOLTAS.NS", "ZYDUSLIFE.NS",
]

# Popular mid-cap / frequently news-mentioned stocks
POPULAR_MIDCAP_TICKERS: list[str] = [
    "ZOMATO.NS", "NYKAA.NS", "DELHIVERY.NS", "POLICYBZR.NS",
    "IRFC.NS", "HAL.NS", "BEL.NS", "NHPC.NS", "RECLTD.NS",
    "PFC.NS", "YESBANK.NS", "FEDERALBNK.NS", "BANDHANBNK.NS",
    "IDFCFIRSTB.NS", "RBLBANK.NS", "MOTHERSON.NS", "ASHOKLEY.NS",
    "BALKRISIND.NS", "PERSISTENT.NS", "COFORGE.NS", "LTTS.NS",
    "KPITTECH.NS", "AUROPHARMA.NS", "ALKEM.NS", "TORNTPOWER.NS",
    "ADANIGREEN.NS", "ADANITRANS.NS", "ATGL.NS", "RVNL.NS",
    "RAILVIKAS.NS", "HUDCO.NS", "SJVN.NS", "NBCC.NS",
]

# Map company aliases to ticker symbols for NER linking
COMPANY_ALIASES: dict[str, str] = {
    # Adani
    "adani enterprises": "ADANIENT.NS",
    "adani": "ADANIENT.NS",
    "adani ports": "ADANIPORTS.NS",
    "adani port": "ADANIPORTS.NS",
    # Apollo
    "apollo hospitals": "APOLLOHOSP.NS",
    "apollo": "APOLLOHOSP.NS",
    # Asian Paints
    "asian paints": "ASIANPAINT.NS",
    "asian paint": "ASIANPAINT.NS",
    # Banks
    "axis bank": "AXISBANK.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfc": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "icici": "ICICIBANK.NS",
    "kotak mahindra bank": "KOTAKBANK.NS",
    "kotak bank": "KOTAKBANK.NS",
    "kotak": "KOTAKBANK.NS",
    "state bank of india": "SBIN.NS",
    "sbi": "SBIN.NS",
    "indusind bank": "INDUSINDBK.NS",
    "indusind": "INDUSINDBK.NS",
    # Bajaj
    "bajaj auto": "BAJAJ-AUTO.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    # Oil & Gas
    "bpcl": "BPCL.NS",
    "bharat petroleum": "BPCL.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS",
    "coal india": "COALINDIA.NS",
    "ongc": "ONGC.NS",
    "oil and natural gas": "ONGC.NS",
    "ntpc": "NTPC.NS",
    "power grid": "POWERGRID.NS",
    "powergrid": "POWERGRID.NS",
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "ril": "RELIANCE.NS",
    # IT
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "tata consultancy services": "TCS.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "hcl technologies": "HCLTECH.NS",
    "hcl tech": "HCLTECH.NS",
    "hcltech": "HCLTECH.NS",
    "wipro": "WIPRO.NS",
    "tech mahindra": "TECHM.NS",
    "ltimindtree": "LTIM.NS",
    "lti mindtree": "LTIM.NS",
    # FMCG
    "hindustan unilever": "HINDUNILVR.NS",
    "hul": "HINDUNILVR.NS",
    "itc": "ITC.NS",
    "nestle india": "NESTLEIND.NS",
    "nestle": "NESTLEIND.NS",
    "britannia": "BRITANNIA.NS",
    "tata consumer": "TATACONSUM.NS",
    "tata consumer products": "TATACONSUM.NS",
    # Pharma
    "sun pharma": "SUNPHARMA.NS",
    "sun pharmaceutical": "SUNPHARMA.NS",
    "dr reddys": "DRREDDY.NS",
    "dr reddy": "DRREDDY.NS",
    "dr. reddy's": "DRREDDY.NS",
    "cipla": "CIPLA.NS",
    "divis laboratories": "DIVISLAB.NS",
    "divis lab": "DIVISLAB.NS",
    # Auto
    "maruti suzuki": "MARUTI.NS",
    "maruti": "MARUTI.NS",
    "tata motors": "TATAMOTORS.NS",
    "mahindra and mahindra": "M&M.NS",
    "mahindra": "M&M.NS",
    "m&m": "M&M.NS",
    "eicher motors": "EICHERMOT.NS",
    "eicher": "EICHERMOT.NS",
    "royal enfield": "EICHERMOT.NS",
    "hero motocorp": "HEROMOTOCO.NS",
    "hero moto": "HEROMOTOCO.NS",
    # Metals
    "tata steel": "TATASTEEL.NS",
    "jsw steel": "JSWSTEEL.NS",
    "hindalco": "HINDALCO.NS",
    # Cement
    "ultratech cement": "ULTRACEMCO.NS",
    "ultratech": "ULTRACEMCO.NS",
    "grasim": "GRASIM.NS",
    "grasim industries": "GRASIM.NS",
    "shree cement": "SHREECEM.NS",
    # Insurance
    "sbi life": "SBILIFE.NS",
    "sbi life insurance": "SBILIFE.NS",
    "hdfc life": "HDFCLIFE.NS",
    "hdfc life insurance": "HDFCLIFE.NS",
    # Others
    "larsen and toubro": "LT.NS",
    "l&t": "LT.NS",
    "titan": "TITAN.NS",
    "titan company": "TITAN.NS",
    "bajaj-auto": "BAJAJ-AUTO.NS",

    # ── NIFTY Next 50 ────────────────────────────────────────────────────────
    # Aditya Birla Capital
    "aditya birla capital": "ABCAPITAL.NS",
    "ab capital": "ABCAPITAL.NS",
    # Ambuja Cements
    "ambuja cements": "AMBUJACEM.NS",
    "ambuja cement": "AMBUJACEM.NS",
    "ambuja": "AMBUJACEM.NS",
    # AU Small Finance Bank
    "au small finance bank": "AUBANK.NS",
    "au bank": "AUBANK.NS",
    "au sfb": "AUBANK.NS",
    # Bank of Baroda
    "bank of baroda": "BANKBARODA.NS",
    # Berger Paints
    "berger paints": "BERGEPAINT.NS",
    "berger": "BERGEPAINT.NS",
    # Biocon
    "biocon": "BIOCON.NS",
    # Bosch
    "bosch": "BOSCHLTD.NS",
    "bosch india": "BOSCHLTD.NS",
    # Canara Bank
    "canara bank": "CANBK.NS",
    "canara": "CANBK.NS",
    # Cholamandalam
    "cholamandalam": "CHOLAFIN.NS",
    "chola finance": "CHOLAFIN.NS",
    "chola": "CHOLAFIN.NS",
    # Colgate
    "colgate": "COLPAL.NS",
    "colgate palmolive": "COLPAL.NS",
    # Container Corporation
    "container corporation": "CONCOR.NS",
    "concor": "CONCOR.NS",
    # Cummins
    "cummins": "CUMMINSIND.NS",
    "cummins india": "CUMMINSIND.NS",
    # DLF
    "dlf": "DLF.NS",
    "dlf limited": "DLF.NS",
    # GAIL
    "gail": "GAIL.NS",
    "gail india": "GAIL.NS",
    # Godrej Consumer
    "godrej consumer": "GODREJCP.NS",
    "godrej consumer products": "GODREJCP.NS",
    "gcpl": "GODREJCP.NS",
    # Godrej Properties
    "godrej properties": "GODREJPROP.NS",
    "godrej prop": "GODREJPROP.NS",
    # Havells
    "havells": "HAVELLS.NS",
    "havells india": "HAVELLS.NS",
    # Hindustan Petroleum
    "hindustan petroleum": "HINDPETRO.NS",
    "hpcl": "HINDPETRO.NS",
    "hp petroleum": "HINDPETRO.NS",
    # ICICI Lombard
    "icici lombard": "ICICIGI.NS",
    "icici general insurance": "ICICIGI.NS",
    # ICICI Prudential
    "icici prudential": "ICICIPRULI.NS",
    "icici pru": "ICICIPRULI.NS",
    # Indraprastha Gas
    "indraprastha gas": "IGL.NS",
    "igl": "IGL.NS",
    # Indus Towers
    "indus towers": "INDUSTOWER.NS",
    "indus tower": "INDUSTOWER.NS",
    # Indian Oil
    "indian oil": "IOC.NS",
    "iocl": "IOC.NS",
    "indian oil corporation": "IOC.NS",
    # IRCTC
    "irctc": "IRCTC.NS",
    "indian railway catering": "IRCTC.NS",
    # Jindal Steel
    "jindal steel": "JINDALSTEL.NS",
    "jindal steel and power": "JINDALSTEL.NS",
    "jspl": "JINDALSTEL.NS",
    # Jubilant Foodworks
    "jubilant foodworks": "JUBLFOOD.NS",
    "jubilant": "JUBLFOOD.NS",
    "dominos": "JUBLFOOD.NS",
    "domino's": "JUBLFOOD.NS",
    # LIC
    "life insurance corporation": "LICI.NS",
    "lic india": "LICI.NS",
    # Lupin
    "lupin": "LUPIN.NS",
    # Marico
    "marico": "MARICO.NS",
    "parachute": "MARICO.NS",
    "saffola": "MARICO.NS",
    # Mphasis
    "mphasis": "MPHASIS.NS",
    # Muthoot Finance
    "muthoot finance": "MUTHOOTFIN.NS",
    "muthoot": "MUTHOOTFIN.NS",
    # Info Edge / Naukri
    "naukri": "NAUKRI.NS",
    "info edge": "NAUKRI.NS",
    "infoedge": "NAUKRI.NS",
    "99acres": "NAUKRI.NS",
    # NMDC
    "nmdc": "NMDC.NS",
    # Page Industries / Jockey
    "page industries": "PAGEIND.NS",
    "jockey": "PAGEIND.NS",
    # Petronet LNG
    "petronet": "PETRONET.NS",
    "petronet lng": "PETRONET.NS",
    # Pidilite
    "pidilite": "PIDILITIND.NS",
    "fevicol": "PIDILITIND.NS",
    "pidilite industries": "PIDILITIND.NS",
    # Punjab National Bank
    "punjab national bank": "PNB.NS",
    "pnb": "PNB.NS",
    # SBI Cards
    "sbi cards": "SBICARD.NS",
    "sbi card": "SBICARD.NS",
    # Siemens
    "siemens": "SIEMENS.NS",
    "siemens india": "SIEMENS.NS",
    # Tata Communications
    "tata communications": "TATACOMM.NS",
    "tatacomm": "TATACOMM.NS",
    # Tata Power
    "tata power": "TATAPOWER.NS",
    # Torrent Pharma
    "torrent pharma": "TORNTPHARM.NS",
    "torrent pharmaceuticals": "TORNTPHARM.NS",
    # Trent / Westside / Zudio
    "trent": "TRENT.NS",
    "westside": "TRENT.NS",
    "zudio": "TRENT.NS",
    # Vedanta
    "vedanta": "VEDL.NS",
    "vedanta resources": "VEDL.NS",
    # Voltas
    "voltas": "VOLTAS.NS",
    # Zydus
    "zydus": "ZYDUSLIFE.NS",
    "zydus lifesciences": "ZYDUSLIFE.NS",
    "cadila": "ZYDUSLIFE.NS",
    "cadila healthcare": "ZYDUSLIFE.NS",

    # ── Popular mid-caps / frequently news-mentioned ──────────────────────────
    # Zomato
    "zomato": "ZOMATO.NS",
    # Nykaa
    "nykaa": "NYKAA.NS",
    "fsnl": "NYKAA.NS",
    # Delhivery
    "delhivery": "DELHIVERY.NS",
    # PolicyBazaar
    "policybazaar": "POLICYBZR.NS",
    "policy bazaar": "POLICYBZR.NS",
    "pb fintech": "POLICYBZR.NS",
    # IRFC
    "irfc": "IRFC.NS",
    "indian railway finance": "IRFC.NS",
    # HAL
    "hindustan aeronautics": "HAL.NS",
    "hal limited": "HAL.NS",
    # BEL
    "bharat electronics": "BEL.NS",
    # NHPC
    "nhpc": "NHPC.NS",
    # REC
    "rec limited": "RECLTD.NS",
    "rural electrification corporation": "RECLTD.NS",
    # PFC
    "power finance corporation": "PFC.NS",
    "power finance corp": "PFC.NS",
    # Yes Bank
    "yes bank": "YESBANK.NS",
    # Federal Bank
    "federal bank": "FEDERALBNK.NS",
    # Bandhan Bank
    "bandhan bank": "BANDHANBNK.NS",
    "bandhan": "BANDHANBNK.NS",
    # IDFC First Bank
    "idfc first bank": "IDFCFIRSTB.NS",
    "idfc first": "IDFCFIRSTB.NS",
    "idfc": "IDFCFIRSTB.NS",
    # RBL Bank
    "rbl bank": "RBLBANK.NS",
    "rbl": "RBLBANK.NS",
    # Motherson
    "motherson": "MOTHERSON.NS",
    "samvardhana motherson": "MOTHERSON.NS",
    # Ashok Leyland
    "ashok leyland": "ASHOKLEY.NS",
    "ashok ley": "ASHOKLEY.NS",
    # Balkrishna Industries
    "balkrishna": "BALKRISIND.NS",
    "bkt tyres": "BALKRISIND.NS",
    # Persistent Systems
    "persistent": "PERSISTENT.NS",
    "persistent systems": "PERSISTENT.NS",
    # Coforge
    "coforge": "COFORGE.NS",
    # L&T Technology Services
    "ltts": "LTTS.NS",
    "l&t technology": "LTTS.NS",
    "lt technology services": "LTTS.NS",
    # KPIT Technologies
    "kpit": "KPITTECH.NS",
    "kpit technologies": "KPITTECH.NS",
    # Aurobindo Pharma
    "aurobindo": "AUROPHARMA.NS",
    "aurobindo pharma": "AUROPHARMA.NS",
    # Alkem Laboratories
    "alkem": "ALKEM.NS",
    "alkem laboratories": "ALKEM.NS",
    # Torrent Power
    "torrent power": "TORNTPOWER.NS",
    # Adani Green
    "adani green": "ADANIGREEN.NS",
    "adani green energy": "ADANIGREEN.NS",
    # Adani Transmission
    "adani transmission": "ADANITRANS.NS",
    "adani energy solutions": "ADANITRANS.NS",
    # Adani Total Gas
    "adani total gas": "ATGL.NS",
    "atgl": "ATGL.NS",
    # RVNL
    "rvnl": "RVNL.NS",
    "rail vikas nigam": "RVNL.NS",
    # HUDCO
    "hudco": "HUDCO.NS",
    "housing urban development": "HUDCO.NS",
    # SJVN
    "sjvn": "SJVN.NS",
    # NBCC
    "nbcc": "NBCC.NS",
    "national buildings construction": "NBCC.NS",
}

# Reverse map: ticker -> canonical name
TICKER_TO_NAME: dict[str, str] = {
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS": "Axis Bank",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "BPCL.NS": "Bharat Petroleum",
    "BHARTIARTL.NS": "Bharti Airtel",
    "BRITANNIA.NS": "Britannia Industries",
    "CIPLA.NS": "Cipla",
    "COALINDIA.NS": "Coal India",
    "DIVISLAB.NS": "Divis Laboratories",
    "DRREDDY.NS": "Dr Reddys Laboratories",
    "EICHERMOT.NS": "Eicher Motors",
    "GRASIM.NS": "Grasim Industries",
    "HCLTECH.NS": "HCL Technologies",
    "HDFCBANK.NS": "HDFC Bank",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "HINDALCO.NS": "Hindalco Industries",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ICICIBANK.NS": "ICICI Bank",
    "INDUSINDBK.NS": "IndusInd Bank",
    "INFY.NS": "Infosys",
    "ITC.NS": "ITC",
    "JSWSTEEL.NS": "JSW Steel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen and Toubro",
    "LTIM.NS": "LTIMindtree",
    "M&M.NS": "Mahindra and Mahindra",
    "MARUTI.NS": "Maruti Suzuki",
    "NESTLEIND.NS": "Nestle India",
    "NTPC.NS": "NTPC",
    "ONGC.NS": "ONGC",
    "POWERGRID.NS": "Power Grid Corporation",
    "RELIANCE.NS": "Reliance Industries",
    "SBILIFE.NS": "SBI Life Insurance",
    "SBIN.NS": "State Bank of India",
    "SHREECEM.NS": "Shree Cement",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TATACONSUM.NS": "Tata Consumer Products",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "TCS.NS": "Tata Consultancy Services",
    "TECHM.NS": "Tech Mahindra",
    "TITAN.NS": "Titan Company",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro",
    # NIFTY Next 50
    "ABCAPITAL.NS":  "Aditya Birla Capital",
    "AMBUJACEM.NS":  "Ambuja Cements",
    "AUBANK.NS":     "AU Small Finance Bank",
    "BANKBARODA.NS": "Bank of Baroda",
    "BERGEPAINT.NS": "Berger Paints",
    "BIOCON.NS":     "Biocon",
    "BOSCHLTD.NS":   "Bosch India",
    "CANBK.NS":      "Canara Bank",
    "CHOLAFIN.NS":   "Cholamandalam Investment",
    "COLPAL.NS":     "Colgate Palmolive India",
    "CONCOR.NS":     "Container Corporation of India",
    "CUMMINSIND.NS": "Cummins India",
    "DLF.NS":        "DLF",
    "GAIL.NS":       "GAIL India",
    "GODREJCP.NS":   "Godrej Consumer Products",
    "GODREJPROP.NS": "Godrej Properties",
    "HAVELLS.NS":    "Havells India",
    "HINDPETRO.NS":  "Hindustan Petroleum",
    "ICICIGI.NS":    "ICICI Lombard General Insurance",
    "ICICIPRULI.NS": "ICICI Prudential Life Insurance",
    "IGL.NS":        "Indraprastha Gas",
    "INDUSTOWER.NS": "Indus Towers",
    "IOC.NS":        "Indian Oil Corporation",
    "IRCTC.NS":      "IRCTC",
    "JINDALSTEL.NS": "Jindal Steel and Power",
    "JUBLFOOD.NS":   "Jubilant Foodworks",
    "LICI.NS":       "Life Insurance Corporation of India",
    "LUPIN.NS":      "Lupin",
    "MARICO.NS":     "Marico",
    "MPHASIS.NS":    "Mphasis",
    "MUTHOOTFIN.NS": "Muthoot Finance",
    "NAUKRI.NS":     "Info Edge (Naukri)",
    "NMDC.NS":       "NMDC",
    "PAGEIND.NS":    "Page Industries",
    "PETRONET.NS":   "Petronet LNG",
    "PIDILITIND.NS": "Pidilite Industries",
    "PNB.NS":        "Punjab National Bank",
    "SBICARD.NS":    "SBI Cards and Payment Services",
    "SIEMENS.NS":    "Siemens India",
    "TATACOMM.NS":   "Tata Communications",
    "TATAPOWER.NS":  "Tata Power",
    "TORNTPHARM.NS": "Torrent Pharmaceuticals",
    "TRENT.NS":      "Trent",
    "VEDL.NS":       "Vedanta",
    "VOLTAS.NS":     "Voltas",
    "ZYDUSLIFE.NS":  "Zydus Lifesciences",
    # Popular mid-caps
    "ZOMATO.NS":     "Zomato",
    "NYKAA.NS":      "Nykaa (FSN E-Commerce)",
    "DELHIVERY.NS":  "Delhivery",
    "POLICYBZR.NS":  "PolicyBazaar (PB Fintech)",
    "IRFC.NS":       "Indian Railway Finance Corporation",
    "HAL.NS":        "Hindustan Aeronautics",
    "BEL.NS":        "Bharat Electronics",
    "NHPC.NS":       "NHPC",
    "RECLTD.NS":     "REC Limited",
    "PFC.NS":        "Power Finance Corporation",
    "YESBANK.NS":    "Yes Bank",
    "FEDERALBNK.NS": "Federal Bank",
    "BANDHANBNK.NS": "Bandhan Bank",
    "IDFCFIRSTB.NS": "IDFC First Bank",
    "RBLBANK.NS":    "RBL Bank",
    "MOTHERSON.NS":  "Samvardhana Motherson International",
    "ASHOKLEY.NS":   "Ashok Leyland",
    "BALKRISIND.NS": "Balkrishna Industries",
    "PERSISTENT.NS": "Persistent Systems",
    "COFORGE.NS":    "Coforge",
    "LTTS.NS":       "L&T Technology Services",
    "KPITTECH.NS":   "KPIT Technologies",
    "AUROPHARMA.NS": "Aurobindo Pharma",
    "ALKEM.NS":      "Alkem Laboratories",
    "TORNTPOWER.NS": "Torrent Power",
    "ADANIGREEN.NS": "Adani Green Energy",
    "ADANITRANS.NS": "Adani Energy Solutions",
    "ATGL.NS":       "Adani Total Gas",
    "RVNL.NS":       "Rail Vikas Nigam",
    "HUDCO.NS":      "HUDCO",
    "SJVN.NS":       "SJVN",
    "NBCC.NS":       "NBCC India",
}

# Sector mapping
SECTOR_MAP: dict[str, list[str]] = {
    "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS"],
    "Banking": [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "SBIN.NS", "INDUSINDBK.NS",
    ],
    "Financial_Services": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "SBILIFE.NS", "HDFCLIFE.NS"],
    "Oil_Gas_Energy": [
        "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
        "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BPCL.NS",
    ],
    "Pharma_Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS"],
    "Automobile": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS",
        "EICHERMOT.NS", "HEROMOTOCO.NS",
    ],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS"],
    "Metals_Mining": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS"],
    "Cement_Construction": ["ULTRACEMCO.NS", "GRASIM.NS", "SHREECEM.NS"],
    "Telecom": ["BHARTIARTL.NS"],
    "Conglomerate": ["LT.NS"],
    "Other": ["TITAN.NS", "ASIANPAINT.NS", "DIVISLAB.NS"],
}

# Keyword categories for news classification
NEWS_KEYWORD_CATEGORIES: dict[str, list[str]] = {
    "earnings": [
        "earnings", "revenue", "profit", "loss", "quarterly results", "annual results",
        "eps", "ebitda", "net income", "operating income", "margin", "guidance",
    ],
    "merger": [
        "merger", "acquisition", "takeover", "buyout", "stake", "deal",
        "partnership", "joint venture", "consolidation",
    ],
    "regulation": [
        "regulation", "regulatory", "sebi", "rbi", "policy", "compliance",
        "ban", "fine", "penalty", "approval", "license",
    ],
    "management": [
        "ceo", "cfo", "director", "board", "appointment", "resignation",
        "leadership", "management change", "restructuring",
    ],
    "dividend": [
        "dividend", "buyback", "bonus", "stock split", "share repurchase",
    ],
    "expansion": [
        "expansion", "new plant", "capacity", "capex", "investment",
        "launch", "new product", "new market", "growth",
    ],
}
