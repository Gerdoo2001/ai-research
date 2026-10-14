param([Parameter(Mandatory=$true)][string]$RepositoryRoot)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($RepositoryRoot)
$qa = Join-Path $root 'model\docs\_ft011_word_qa\word_pdf'
New-Item -ItemType Directory -Path $qa -Force | Out-Null
$files = @(
    'model\docs\fine_tuning\FT011_Final_Report_20261013.docx',
    'model\docs\fine_tuning\DenseNet201_VIRIDIS_GE_CLAHE_Fine_Tuning_Report.docx',
    'model\docs\Pneumonia_Three_Class_Classification_Full_Journal_20261013.docx'
)
# This creates a private automation instance. Do not attach to the user's Word window.
$word = New-Object -ComObject Word.Application
$initialDocuments = $word.Documents.Count
$owned = ($initialDocuments -eq 0)
if (-not $owned) { throw 'Word returned an existing document session; refusing to alter or close it.' }
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    foreach ($relative in $files) {
        $inputFile = Join-Path $root $relative
        $pdf = Join-Path $qa ([IO.Path]::GetFileNameWithoutExtension($inputFile) + '.pdf')
        Write-Output "Exporting $inputFile"
        $doc = $word.Documents.Open($inputFile, $false, $true)
        try {
            $doc.Repaginate()
            $pages = $doc.ComputeStatistics(2)
            $doc.ExportAsFixedFormat($pdf, 17)
            Write-Output "Exported $pages pages to $pdf"
        }
        finally { $doc.Close(0) }
    }
}
finally {
    if ($owned) { $word.Quit() }
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
}
