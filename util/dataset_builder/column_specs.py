"""Column selections and merge-step specifications for the full dataset build.

The column lists mirror GetFMPData/ConstructFullData.ipynb exactly so the
reconstructed dataset keeps the original value-column names. Each fundamental
table additionally retains its report period end date as `{prefix}PeriodEnd`
(the original notebook lost it because `merge_asof` consumed `date` as the
join key).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

S3_BUCKET = "sagemaker-us-east-1-209479286572"
S3_PREFIX = "datasets/QuantTradingModelData"
S3_REGION = "us-east-1"

PRICE_DATATYPES = ("adjusteddailyprice", "unadjusteddailyprice")

INCM_COLS = [
    'date', 'symbol', 'filingDate', 'acceptedDate',
    'revenue', 'costOfRevenue', 'grossProfit',
    'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses',
    'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses',
    'otherExpenses', 'operatingExpenses', 'costAndExpenses',
    'netInterestIncome', 'interestIncome', 'interestExpense',
    'depreciationAndAmortization', 'ebitda', 'ebit',
    'nonOperatingIncomeExcludingInterest', 'operatingIncome',
    'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeTaxExpense',
    'netIncomeFromContinuingOperations', 'netIncomeFromDiscontinuedOperations',
    'otherAdjustmentsToNetIncome', 'netIncome', 'netIncomeDeductions',
    'bottomLineNetIncome', 'eps', 'epsDiluted',
    'weightedAverageShsOut', 'weightedAverageShsOutDil',
]

BAL_COLS = [
    'date', 'symbol', 'filingDate', 'acceptedDate',
    'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments',
    'netReceivables', 'accountsReceivables', 'otherReceivables',
    'inventory', 'prepaids', 'otherCurrentAssets', 'totalCurrentAssets',
    'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets',
    'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets',
    'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets',
    'totalPayables', 'accountPayables', 'otherPayables', 'accruedExpenses',
    'shortTermDebt', 'capitalLeaseObligationsCurrent', 'taxPayables',
    'deferredRevenue', 'otherCurrentLiabilities', 'totalCurrentLiabilities',
    'longTermDebt', 'capitalLeaseObligationsNonCurrent', 'deferredRevenueNonCurrent',
    'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities',
    'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations',
    'totalLiabilities', 'treasuryStock', 'preferredStock', 'commonStock',
    'retainedEarnings', 'additionalPaidInCapital',
    'accumulatedOtherComprehensiveIncomeLoss', 'otherTotalStockholdersEquity',
    'totalStockholdersEquity', 'totalEquity', 'minorityInterest',
    'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt',
]

CF_COLS = [
    'date', 'symbol', 'filingDate', 'acceptedDate',
    'netIncome', 'depreciationAndAmortization', 'deferredIncomeTax',
    'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables',
    'inventory', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems',
    'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment',
    'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments',
    'otherInvestingActivities', 'netCashProvidedByInvestingActivities',
    'netDebtIssuance', 'longTermNetDebtIssuance', 'shortTermNetDebtIssuance',
    'netStockIssuance', 'netCommonStockIssuance', 'commonStockIssuance',
    'commonStockRepurchased', 'netPreferredStockIssuance', 'netDividendsPaid',
    'commonDividendsPaid', 'preferredDividendsPaid', 'otherFinancingActivities',
    'netCashProvidedByFinancingActivities', 'effectOfForexChangesOnCash',
    'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod',
    'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow',
    'incomeTaxesPaid', 'interestPaid',
]

KM_COLS = [
    'date', 'symbol',
    'marketCap', 'enterpriseValue', 'evToSales', 'evToOperatingCashFlow',
    'evToFreeCashFlow', 'evToEBITDA', 'netDebtToEBITDA', 'currentRatio',
    'incomeQuality', 'grahamNetNet', 'taxBurden', 'interestBurden',
    'workingCapital', 'investedCapital', 'returnOnAssets',
    'operatingReturnOnAssets', 'returnOnTangibleAssets', 'returnOnEquity',
    'returnOnInvestedCapital', 'returnOnCapitalEmployed', 'earningsYield',
    'freeCashFlowYield', 'capexToOperatingCashFlow', 'capexToDepreciation',
    'capexToRevenue', 'salesGeneralAndAdministrativeToRevenue',
    'researchAndDevelopementToRevenue', 'stockBasedCompensationToRevenue',
    'intangiblesToTotalAssets', 'averageReceivables', 'averagePayables',
    'averageInventory', 'daysOfSalesOutstanding', 'daysOfPayablesOutstanding',
    'daysOfInventoryOutstanding', 'operatingCycle', 'cashConversionCycle',
    'freeCashFlowToEquity', 'freeCashFlowToFirm', 'tangibleAssetValue',
    'netCurrentAssetValue', 'grahamNumber',
]

EV_COLS = [
    'date', 'symbol',
    'stockPrice', 'numberOfShares', 'marketCapitalization',
    'minusCashAndCashEquivalents', 'addTotalDebt', 'enterpriseValue',
]

PROFILE_COLS = ['symbol', 'exchange', 'industry', 'sector', 'ipoDate']


@dataclass
class FundamentalMergeSpec:
    """One quarterly table to be as-of merged onto the daily spine.

    pit_key:        column holding the public availability date (None ->
                    merge on report period end date, the original behavior).
    borrow_pit:     if True, the availability date is borrowed from the
                    income statement filingDate via (symbol, period end).
    left_renames:   applied to the accumulated panel BEFORE this merge to
                    resolve cross-statement column collisions (matches the
                    original notebook's rename sequence).
    right_renames:  applied to the quarterly table before the merge
                    (originally applied after; equivalent once collisions
                    are resolved).
    """
    name: str
    datatype: str
    prefix: str
    columns: List[str]
    pit_key: Optional[str] = 'filingDate'
    borrow_pit: bool = False
    left_renames: Dict[str, str] = field(default_factory=dict)
    right_renames: Dict[str, str] = field(default_factory=dict)


MERGE_SPECS = [
    FundamentalMergeSpec(
        name='enterprise values', datatype='enterprisevalues', prefix='ev',
        columns=EV_COLS,
        pit_key=None,  # market-observable quantities; merged on period end
    ),
    FundamentalMergeSpec(
        name='income statement', datatype='incomestatement', prefix='incm',
        columns=INCM_COLS,
        right_renames={'filingDate': 'incmFilingDate', 'acceptedDate': 'incmAcceptedDate'},
    ),
    FundamentalMergeSpec(
        name='balance sheet', datatype='balancesheet', prefix='bal',
        columns=BAL_COLS,
        right_renames={'filingDate': 'balFilingDate', 'acceptedDate': 'balAcceptedDate'},
    ),
    FundamentalMergeSpec(
        name='cash flow', datatype='cashflow', prefix='cf',
        columns=CF_COLS,
        left_renames={
            'netIncome': 'incmNetIncome',
            'depreciationAndAmortization': 'incmDepreciationAndAmortization',
            'accountsReceivables': 'balAccountsReceivables',
            'inventory': 'balInventory',
        },
        right_renames={
            'filingDate': 'cfFilingDate', 'acceptedDate': 'cfAcceptedDate',
            'netIncome': 'cfNetIncome',
            'depreciationAndAmortization': 'cfDepreciationAndAmortization',
            'accountsReceivables': 'cfAccountsReceivables',
            'inventory': 'cfInventory',
        },
    ),
    FundamentalMergeSpec(
        name='key metrics', datatype='keymetrics', prefix='km',
        columns=KM_COLS,
        borrow_pit=True,  # no filingDate of its own; borrowed from income statement
        left_renames={'enterpriseValue': 'evEnterpriseValue'},
        right_renames={'enterpriseValue': 'kmEnterpriseValue', 'filingDate': 'kmFilingDate'},
    ),
]
