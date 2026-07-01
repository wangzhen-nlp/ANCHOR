import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  exportJsonlFile,
  exportPerFileDirectory,
  loadArtifactTool,
} from "./export_per_file_alarm_xlsx.mjs";


const sampleGroup = {
  uuid: "GROUP-1",
  alarms: [
    {
      告警编码ID: "A-2",
      告警标题: "链路中断",
      告警源: "NE-2",
      告警首次发生时间: "2026-07-01 08:03:00",
    },
    {
      告警编码ID: "A-1",
      告警标题: "网元断连",
      告警源: "NE-1",
      告警首次发生时间: "2026-07-01 08:00:00",
      告警清除时间: "2026-07-01 08:10:00",
      扩展信息: { port: "1/1" },
    },
    {
      告警编码ID: "A-4",
      告警标题: "设备离线",
      告警源: "NE-1",
      告警首次发生时间: "2026-07-01 08:01:00",
      告警最后发生时间: "2026-07-01 08:09:00",
    },
    { 告警编码ID: "A-3", 告警标题: "电源告警", 告警源: "NE-3" },
  ],
  symptoms: [{ eid: "A-1", alarm: "网元断连", alarm_source: "NE-1" }],
  ne_info: {
    "NE-1": { site_id: "SITE-1", manufacturer: "HUAWEI", site_name: "站点一" },
    "NE-2": { site_id: "SITE-1", manufacturer: "HUAWEI", site_name: "站点一" },
    "NE-3": { site_id: "SITE-2", manufacturer: "ZTE", site_name: "站点二" },
  },
};


test("exports sorted hierarchical alarm workbook without duplicate visual alarms", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "alarm-xlsx-"));
  const inputPath = path.join(tempDir, "GROUP-1.jsonl");
  const outputPath = path.join(tempDir, "GROUP-1.xlsx");
  await fs.writeFile(inputPath, `${JSON.stringify(sampleGroup)}\n`, "utf8");
  assert.equal(await exportJsonlFile(inputPath, outputPath), 4);

  const { FileBlob, SpreadsheetFile } = await loadArtifactTool();
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  const inspected = await workbook.inspect({
    kind: "table",
    range: "告警明细!A1:H5",
    include: "values,formulas",
    tableMaxRows: 5,
    tableMaxCols: 8,
  });
  const output = inspected.ndjson;
  assert.match(output, /站点ID/);
  assert.match(output, /设备厂家/);
  assert.match(output, /告警源/);
  assert.match(output, /SITE-1/);
  assert.match(output, /NE-1/);
  assert.match(output, /NE-2/);
  assert.match(output, /SITE-2/);
});


test("writes one same-name xlsx for every per-file jsonl", async () => {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "alarm-xlsx-dir-"));
  const inputDir = path.join(tempDir, "jsonl");
  const outputDir = path.join(tempDir, "xlsx");
  await fs.mkdir(inputDir);
  await fs.writeFile(path.join(inputDir, "GROUP-1.jsonl"), `${JSON.stringify(sampleGroup)}\n`, "utf8");
  const stats = await exportPerFileDirectory(inputDir, outputDir);
  assert.equal(stats.input_file_count, 1);
  assert.equal(stats.output_file_count, 1);
  assert.equal(stats.alarm_count, 4);
  await fs.access(path.join(outputDir, "GROUP-1.xlsx"));
});
