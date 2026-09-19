import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const sourcePath = "data/match_day_results/match_results_master.csv";
const outputDir = "outputs/season_match_capture";
const outputPath = `${outputDir}/season_matches_by_season.xlsx`;
const previewPath = `${outputDir}/season_matches_preview.png`;
const sourcePreviewPath = `${outputDir}/captured_source_preview.png`;

function parseCsvLine(line) {
  const fields = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') {
        field += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (ch === "," && !quoted) {
      fields.push(field);
      field = "";
    } else {
      field += ch;
    }
  }
  fields.push(field);
  return fields;
}

const csvText = await fs.readFile(sourcePath, "utf8");
const lines = csvText.replace(/^\uFEFF/, "").trimEnd().split(/\r?\n/);
const headers = parseCsvLine(lines[0]);
const seasonIndex = headers.indexOf("season_id");
if (seasonIndex < 0) throw new Error("The source CSV does not contain season_id.");

const counts = new Map();
for (const line of lines.slice(1)) {
  if (!line) continue;
  const fields = parseCsvLine(line);
  const season = Number(fields[seasonIndex]);
  if (!Number.isInteger(season)) continue;
  counts.set(season, (counts.get(season) ?? 0) + 1);
}

const captured = [...counts.entries()].sort((a, b) => a[0] - b[0]);
const minSeason = captured[0][0];
const maxSeason = captured[captured.length - 1][0];
const allSeasons = Array.from(
  { length: maxSeason - minSeason + 1 },
  (_, index) => minSeason + index
);
const missingSeasons = allSeasons.filter((season) => !counts.has(season));

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Season chart");
const source = workbook.worksheets.add("Captured source");
summary.showGridLines = false;
source.showGridLines = false;
summary.tabColor = "#1F4E78";
source.tabColor = "#9EADBA";

const fontFamily = "Arial";
const darkBlue = "#1F4E78";
const blue = "#2F75B5";
const paleBlue = "#D9EAF7";
const paleRed = "#FDE9E7";
const red = "#C00000";
const gray = "#667085";

summary.getRange("A2:T2").format.borders = {
  bottom: { style: "thin", color: darkBlue },
};
summary.getRange("A2").values = [["Matches captured by season"]];
summary.getRange("A2").format.font = {
  name: fontFamily,
  size: 16,
  bold: true,
  color: "#172B4D",
};
summary.getRange("A3").values = [[
  `Continuous season range ${minSeason}–${maxSeason}. Seasons with no source rows are shown as zero and marked Not captured.`,
]];
summary.getRange("A3:T3").format.font = {
  name: fontFamily,
  size: 10,
  italic: true,
  color: gray,
};

summary.getRange("A5:H6").values = [
  ["Lowest season", null, "Highest season", null, "Captured seasons", null, "Missing seasons", null],
  [minSeason, null, maxSeason, null, captured.length, null, missingSeasons.length, null],
];
for (const pair of [["A5:B6"], ["C5:D6"], ["E5:F6"], ["G5:H6"]]) {
  const range = summary.getRange(pair[0]);
  range.format.borders = { preset: "outside", style: "thin", color: "#B4C7E7" };
}
summary.getRange("A5:H5").format = {
  fill: paleBlue,
  font: { name: fontFamily, size: 10, bold: true, color: darkBlue },
  verticalAlignment: "center",
};
summary.getRange("A6:H6").format.font = {
  name: fontFamily,
  size: 14,
  bold: true,
  color: "#172B4D",
};

summary.getRange("A8:C8").values = [["Season", "Matches captured", "Capture status"]];
summary.getRange("A8:C8").format = {
  fill: darkBlue,
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: {
    insideVertical: { style: "thin", color: "#FFFFFF" },
  },
};
summary.getRange(`A9:A${8 + allSeasons.length}`).values = allSeasons.map((season) => [season]);
const summaryEndRow = 8 + allSeasons.length;
const sourceStartRow = 5;
const sourceEndRow = sourceStartRow + captured.length - 1;
summary.getRange("B9").formulas = [[
  `=SUMIFS('Captured source'!$B$${sourceStartRow}:$B$${sourceEndRow},'Captured source'!$A$${sourceStartRow}:$A$${sourceEndRow},A9)`,
]];
summary.getRange(`B9:B${summaryEndRow}`).fillDown();
summary.getRange("C9").formulas = [["=IF(B9=0,\"Not captured\",\"Captured\")"]];
summary.getRange(`C9:C${summaryEndRow}`).fillDown();
summary.getRange(`A9:C${summaryEndRow}`).format.font = { name: fontFamily, size: 10, color: "#222222" };
summary.getRange(`A9:B${summaryEndRow}`).format.numberFormat = "#,##0";
summary.getRange(`A9:C${summaryEndRow}`).format.verticalAlignment = "center";
summary.getRange(`A9:A${summaryEndRow}`).format.horizontalAlignment = "right";
summary.getRange(`B9:B${summaryEndRow}`).format.horizontalAlignment = "right";
summary.getRange(`C9:C${summaryEndRow}`).format.horizontalAlignment = "center";
summary.getRange(`A9:C${summaryEndRow}`).conditionalFormats.addCustom("=$B9=0", {
  fill: paleRed,
  font: { color: red, bold: true },
});
summary.freezePanes.freezeRows(8);

const chart = summary.charts.add("line", summary.getRange(`A8:B${summaryEndRow}`));
chart.setPosition("E8", "T31");
chart.title = "Total matches captured per season";
chart.titleTextStyle.fontSize = 14;
chart.titleTextStyle.typeface = fontFamily;
chart.hasLegend = false;
chart.xAxis = {
  axisType: "textAxis",
  textStyle: { typeface: fontFamily, fontSize: 9 },
  tickLabelInterval: 25,
};
chart.yAxis = {
  numberFormatCode: "#,##0",
  numberFormatSourceLinked: false,
  textStyle: { typeface: fontFamily, fontSize: 10 },
};
chart.xAxis.title.text = "Season";
chart.yAxis.title.text = "Total matches captured";
if (chart.series.items.length > 0) {
  chart.series.items[0].line = { fill: blue, style: "solid", width: 2 };
}

summary.getRange("E34:I34").format.borders = {
  bottom: { style: "thin", color: darkBlue },
};
summary.getRange("E34").values = [[`Missing seasons (${missingSeasons.length})`]];
summary.getRange("E34").format.font = {
  name: fontFamily,
  size: 12,
  bold: true,
  color: "#172B4D",
};
summary.getRange("E35:I35").values = [["Season ID", "Season ID", "Season ID", "Season ID", "Season ID"]];
summary.getRange("E35:I35").format = {
  fill: "#C00000",
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
};
const missingRows = Math.ceil(missingSeasons.length / 5);
const missingMatrix = Array.from({ length: missingRows }, (_, row) =>
  Array.from({ length: 5 }, (_, col) => missingSeasons[col * missingRows + row] ?? null)
);
summary.getRange(`E36:I${35 + missingRows}`).values = missingMatrix;
summary.getRange(`E36:I${35 + missingRows}`).format = {
  fill: paleRed,
  font: { name: fontFamily, size: 10, color: red },
  horizontalAlignment: "right",
  numberFormat: "0",
};

summary.getRange("A:T").format.font = { name: fontFamily };
summary.getRange("A:A").format.columnWidth = 14;
summary.getRange("B:B").format.columnWidth = 19;
summary.getRange("C:C").format.columnWidth = 17;
summary.getRange("D:D").format.columnWidth = 3;
summary.getRange("E:I").format.columnWidth = 14;
summary.getRange("J:T").format.columnWidth = 11;
summary.getRange("2:2").format.rowHeight = 24;
summary.getRange("8:8").format.rowHeight = 22;

source.getRange("A2:B2").format.borders = {
  bottom: { style: "thin", color: darkBlue },
};
source.getRange("A2").values = [["Captured matches by source season"]];
source.getRange("A2").format.font = {
  name: fontFamily,
  size: 14,
  bold: true,
  color: "#172B4D",
};
source.getRange("A3").values = [[`Source: match_results_master.csv (${lines.length - 1} match rows)`]];
source.getRange("A3:B3").format.font = {
  name: fontFamily,
  size: 10,
  italic: true,
  color: gray,
};
source.getRange("A4:B4").values = [["Season", "Matches captured"]];
source.getRange("A4:B4").format = {
  fill: darkBlue,
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  borders: { insideVertical: { style: "thin", color: "#FFFFFF" } },
};
source.getRange(`A${sourceStartRow}:B${sourceEndRow}`).values = captured;
source.getRange(`A${sourceStartRow}:B${sourceEndRow}`).format = {
  font: { name: fontFamily, size: 10, color: "#222222" },
  numberFormat: "#,##0",
  horizontalAlignment: "right",
  verticalAlignment: "center",
};
source.getRange("A:A").format.columnWidth = 24;
source.getRange("B:B").format.columnWidth = 30;
source.freezePanes.freezeRows(4);

workbook.recalculate();

const summaryCheck = await workbook.inspect({
  kind: "table",
  range: "Season chart!A2:I16",
  include: "values,formulas",
  tableMaxRows: 16,
  tableMaxCols: 9,
});
console.log("SUMMARY_CHECK");
console.log(summaryCheck.ndjson);

const missingCheck = await workbook.inspect({
  kind: "table",
  range: `Season chart!E34:I${35 + missingRows}`,
  include: "values,formulas",
  tableMaxRows: 60,
  tableMaxCols: 5,
});
console.log("MISSING_CHECK");
console.log(missingCheck.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log("ERROR_CHECK");
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({
  sheetName: "Season chart",
  range: "A1:T82",
  scale: 1.25,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const sourcePreview = await workbook.render({
  sheetName: "Captured source",
  range: "A1:B30",
  scale: 1.25,
  format: "png",
});
await fs.writeFile(sourcePreviewPath, new Uint8Array(await sourcePreview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const exported = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const exportedDrawings = await exported.inspect({
  kind: "drawing",
  sheetId: "Season chart",
  maxChars: 5000,
});
console.log("EXPORTED_DRAWINGS");
console.log(exportedDrawings.ndjson);
console.log(JSON.stringify({
  outputPath,
  previewPath,
  minSeason,
  maxSeason,
  capturedSeasons: captured.length,
  missingSeasons: missingSeasons.length,
  totalMatches: lines.length - 1,
}, null, 2));


