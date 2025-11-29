NOISE_PATTERNS = [
    # --- NEW: store header + address-ish lines (stop these becoming items) ---
    r"^\s*publix\b",  # "Publix" / "Publix."
    r"\bshopping\s+center\b",  # "Hollieanna shopping center"
    # street address with multiple words before suffix ("741 south orlando avenue")
    r"^\s*\d{1,6}\s+[a-z0-9\s]+\b(ave|avenue|st|street|rd|road|blvd|boulevard|drive|dr|ln|lane|pkwy|parkway|hwy|highway)\b",
    r"^\s*[a-z\s]+,\s*[a-z]{2}\s*$",  # "Winter Park, FL" (if comma shows up)
    r"^\s*[a-z\s]+\s+[a-z]{2}\s*$",   # "Winter Park FL"
    r"^\s*[a-z\s]+\s+[a-z]{2}\s+\d{5}(-\d{4})?\s*$",  # "City ST 12345"

    # --- NEW: promo/header words that sneak in ---
    r"^\s*promotion\b",

    # --- NEW: price+flag lines like "13.99 F" / "3.79 E" ---
    r"^\s*\$?\d{1,6}([.,]\d{2})?\s*[a-z]\s*$",

    # --- existing patterns (unchanged) ---
    r"^apply\b", r"jobs\b", r"publix\.?jobs", r"\bcareer\b",
    r"^payment", r"^entry\s+method", r"^you\s?saved\b", r"^savings\b",
    r"^acct[:# ]", r"^account\b", r"^trace[:# ]", r"\bapproval\b", r"\bauth\b",
    r"\bsubtotal\b", r"\bsub\s*total\b", r"\btax\b", r"\btotal\b", r"\bchange\b", r"\bbalance\b",
    r"\bdebit\b", r"\bcredit\b", r"\bvisa\b", r"\bmastercard\b", r"\bdiscover\b", r"\bamex\b",
    r"\bthank you\b|\bthanks\b", r"\bcashier\b|\bmanager\b|\bstore\b\s*#",
    r"\bpresto!?$", r"\bpresto!\b", r"\bplaza\b|\bmall\b|\bmillenia\b",
    r"https?://", r"\bwww\.", r"@[A-Za-z0-9_]+",
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    r"\b\d{1,4}\s+[A-Za-z0-9]+\s+(ave|avenue|st|street|rd|road|blvd|drive|dr)\b",
    r"\btrace\s*#?:?\s*\d+\b", r"\bacct\s*#?:?\s*\w+\b",
    r"\bcontactless\b|\btap\b|\bchip\b", r"\bmerchant\b|\bterminal\b|\bpos\b",
    r"\bgrocery\s?item\b$",
]
