"""Deterministic sample uploads for the offline workbench demonstration.

Run ``python -m simulator.export_samples`` to rewrite ``data/sample_uploads/``
byte-for-byte. The transaction export deliberately mimics a messy bank
extract: alternate headers, a small mule chain and five malformed/duplicate
rows so the ingestion audit tab always has something real to show.

No clock, no network and no randomness: the files are identical on every run.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from openpyxl import Workbook

TRANSACTIONS_FILE: str = "transactions_messy_bank_export.csv"
ACCOUNTS_FILE: str = "account_master_kyc.xlsx"
COMPLAINTS_FILE: str = "ncrp_complaints_interstate.csv"

_FIXED_TIMESTAMP: datetime = datetime(2024, 6, 1, 0, 0, 0)
_ZIP_STAMP: tuple[int, int, int, int, int, int] = (2024, 6, 1, 0, 0, 0)

_TXN_HEADER: str = "UTR,Debit_Acc,PAYEE,Amount_INR,Txn_Time,Mode\n"

# Ten valid legs: fan-in to MULE-001, rapid relay to a terminal, then a
# second collection episode. Amounts decay at each hop, as in a real mule chain.
_TXN_ROWS: tuple[str, ...] = (
    "DEMO-T0001,VICTIM-001,MULE-001,25000.00,2024-06-01 09:15:00,UPI\n",
    "DEMO-T0002,VICTIM-002,MULE-001,18000.00,2024-06-01 09:22:00,UPI\n",
    "DEMO-T0003,VICTIM-003,MULE-001,12000.00,2024-06-01 09:31:00,IMPS\n",
    "DEMO-T0004,MULE-001,MULE-002,52000.00,2024-06-01 09:40:00,UPI\n",
    "DEMO-T0005,MULE-002,MULE-003,49000.00,2024-06-01 09:48:00,UPI\n",
    "DEMO-T0006,MULE-003,TERM-001,47000.00,2024-06-01 09:55:00,AEPS\n",
    "DEMO-T0007,VICTIM-004,MULE-002,9000.00,2024-06-01 10:05:00,UPI\n",
    "DEMO-T0008,MULE-002,MULE-003,8500.00,2024-06-01 10:12:00,UPI\n",
    "DEMO-T0009,MULE-003,TERM-001,8000.00,2024-06-01 10:20:00,AEPS\n",
    "DEMO-T0010,VICTIM-005,MULE-001,7000.00,2024-06-01 10:30:00,UPI\n",
)

# One row per rejection reason the ingester can produce for the ledger.
_AUDIT_ROWS: tuple[str, ...] = (
    ("AUDIT-BAD-AMOUNT,VICTIM-006,MULE-001,not-a-number,"
     "2024-06-01 10:40:00,UPI\n"),
    ",VICTIM-007,MULE-001,4000.00,2024-06-01 10:45:00,UPI\n",
    "DEMO-T0001,VICTIM-008,MULE-002,3000.00,2024-06-01 10:50:00,UPI\n",
    "AUDIT-NEGATIVE,VICTIM-009,MULE-002,-500.00,2024-06-01 10:55:00,UPI\n",
    "AUDIT-BAD-TIME,VICTIM-010,MULE-003,2000.00,not-a-timestamp,UPI\n",
)

_COMPLAINT_HEADER: str = (
    "ACK,Victim_Acc,Suspect_Acc,Fraud_Category,Reported_On,"
    "Cheated_Amount,Disputed_UTR,State\n"
)

_COMPLAINT_ROWS: tuple[str, ...] = (
    ("NCRP-BR-2024-0001,VICTIM-001,MULE-001,UPI_FRAUD,"
     "2024-06-02 11:00:00,25000.00,DEMO-T0001,Bihar\n"),
    ("NCRP-JH-2024-0002,VICTIM-002,MULE-001,UPI_FRAUD,"
     "2024-06-03 09:30:00,18000.00,DEMO-T0002,Jharkhand\n"),
    ("NCRP-MH-2024-0003,VICTIM-003,MULE-001,IMPS_FRAUD,"
     "2024-06-04 15:45:00,12000.00,DEMO-T0003,Maharashtra\n"),
)

_ACCOUNT_HEADER: tuple[str, ...] = (
    "account_no", "ifsc", "account_opening_date", "branch_code",
    "mobile_no", "imei", "kyc_pincode", "account_type",
)

# MULE-001/002/003 share a branch and open within 72 hours; MULE-001/002
# also share a handset. Both are genuine Phase 2 collusion signals.
_ACCOUNT_ROWS: tuple[tuple[str, ...], ...] = (
    ("VICTIM-001", "UTIB0000123", "2019-04-02", "BR010",
     "9876500001", "351104510000001", "800001", "SAVINGS"),
    ("VICTIM-002", "UTIB0000456", "2020-08-11", "BR011",
     "9876500002", "351104510000002", "800002", "SAVINGS"),
    ("VICTIM-003", "SBIN0000123", "2018-01-19", "BR012",
     "9876500003", "351104510000003", "800003", "SAVINGS"),
    ("MULE-001", "UTIB0000999", "2024-05-20", "BR777",
     "9876500004", "351104510123456", "800013", "PMJDY"),
    ("MULE-002", "UTIB0000999", "2024-05-21", "BR777",
     "9876500005", "351104510123456", "800013", "PMJDY"),
    ("MULE-003", "UTIB0000999", "2024-05-22", "BR777",
     "9876500006", "351104510999999", "800013", "SAVINGS"),
    ("TERM-001", "UTIB0000777", "2022-03-10", "BR077",
     "9876500007", "351104510998871", "800013", "CURRENT"),
)


def _stabilise_xlsx(path: Path) -> None:
    """Rewrite the archive with fixed content and entry timestamps.

    openpyxl stamps ``dcterms:modified`` with the wall clock during save, so the
    core-properties XML is normalised here as well as the zip metadata.
    """
    with ZipFile(path) as source:
        entries = [
            (info.filename, source.read(info.filename))
            for info in source.infolist()
        ]
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in entries:
            if name == "docProps/core.xml":
                payload = re.sub(
                    rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)",
                    rb"\g<1>2024-06-01T00:00:00Z\g<2>",
                    payload,
                )
            info = ZipInfo(name, date_time=_ZIP_STAMP)
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, payload)


def _write_accounts_xlsx(path: Path) -> None:
    workbook = Workbook()
    workbook.properties.created = _FIXED_TIMESTAMP
    workbook.properties.modified = _FIXED_TIMESTAMP
    sheet = workbook.active
    sheet.title = "AccountMaster"
    sheet.append(list(_ACCOUNT_HEADER))
    for row in _ACCOUNT_ROWS:
        sheet.append(list(row))
    workbook.save(path)
    _stabilise_xlsx(path)


def write_sample_uploads(
    directory: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write the three demo files; returns their paths in upload order."""
    target = (
        directory
        if directory is not None
        else Path(__file__).resolve().parents[2] / "data" / "sample_uploads"
    )
    target.mkdir(parents=True, exist_ok=True)
    transactions = target / TRANSACTIONS_FILE
    accounts = target / ACCOUNTS_FILE
    complaints = target / COMPLAINTS_FILE

    transactions.write_text(
        _TXN_HEADER + "".join(_TXN_ROWS) + "".join(_AUDIT_ROWS),
        encoding="utf-8",
        newline="",
    )
    _write_accounts_xlsx(accounts)
    complaints.write_text(
        _COMPLAINT_HEADER + "".join(_COMPLAINT_ROWS),
        encoding="utf-8",
        newline="",
    )
    return transactions, accounts, complaints


def main() -> int:
    paths = write_sample_uploads()
    print("Wrote sample uploads:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
