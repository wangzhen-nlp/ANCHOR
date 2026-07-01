#!/usr/bin/env node
/** 把 complete_group_topology.py --per-file 的 JSONL 逐个导出为层级告警 Excel。 */

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";


const PRIMARY_COLUMNS = [
  "站点ID",
  "设备厂家",
  "告警源",
  "告警标题",
  "告警首次发生时间",
  "告警清除时间",
  "告警最后发生时间",
];

const OTHER_MAIN_COLUMNS = [
  "故障组ID",
  "告警序号",
  "告警编码ID",
  "告警标准名",
  "设备名称",
  "设备域",
  "站点名称",
  "站点类型",
  "区域ID",
  "告警级别",
  "告警状态",
  "工单号",
  "经度",
  "纬度",
];

const MAIN_COLUMNS = [...PRIMARY_COLUMNS, ...OTHER_MAIN_COLUMNS];

const FIELD_ALIASES = {
  告警编码ID: ["告警编码ID", "告警ID", "eid", "alarm_id", "event_id", "id"],
  告警标题: ["告警标题", "alarm", "alarm_type", "alarm_title", "title"],
  告警标准名: ["告警标准名", "告警标准化名称", "standard_alarm_name", "standard_name"],
  告警源: ["告警源", "alarm_source", "ne_id", "source"],
  设备名称: ["设备名称", "网元名称", "device_name", "ne_name"],
  设备域: ["设备域", "domain", "网络专业", "告警源专业", "专业"],
  站点ID: ["站点ID", "site_id", "node", "site"],
  站点名称: ["站点名称", "site_name"],
  站点类型: ["站点类型", "site_type"],
  区域ID: ["区域ID", "region_id"],
  告警级别: ["告警级别", "告警等级", "级别", "severity"],
  告警状态: ["告警状态", "状态", "alarm_status", "status"],
  告警首次发生时间: [
    "告警首次发生时间",
    "告警发生时间",
    "首次发生时间",
    "发生时间",
    "alarm_time",
    "time",
    "ts",
  ],
  告警清除时间: ["告警清除时间", "清除时间", "alarm_clear_time", "clear_time"],
  告警最后发生时间: ["告警最后发生时间", "最后发生时间", "last_occurrence_time"],
  工单号: ["工单号", "ticket_id", "work_order_id"],
  设备厂家: ["设备厂家", "设备厂家名称", "厂家", "manufacturer"],
  经度: ["经度", "longitude", "lon", "lng"],
  纬度: ["纬度", "latitude", "lat"],
};

const CONSUMED_ALARM_FIELDS = new Set(Object.values(FIELD_ALIASES).flat());


export async function loadArtifactTool() {
  try {
    return await import("@oai/artifact-tool");
  } catch (originalError) {
    const nodeModules =
      process.env.ARTIFACT_TOOL_NODE_MODULES ||
      path.join(
        os.homedir(),
        ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules",
      );
    const require = createRequire(import.meta.url);
    try {
      const resolved = require.resolve("@oai/artifact-tool", { paths: [nodeModules] });
      return import(pathToFileURL(resolved).href);
    } catch (fallbackError) {
      throw new Error(
        `找不到 @oai/artifact-tool；已检查项目依赖和 Codex 自带运行时: ${nodeModules}`,
        { cause: fallbackError || originalError },
      );
    }
  }
}


function normalizeText(value) {
  return value === null || value === undefined ? "" : String(value).trim();
}


function firstValue(record, fields) {
  if (!record || typeof record !== "object" || Array.isArray(record)) return "";
  for (const field of fields) {
    const value = record[field];
    if (value !== null && value !== undefined && value !== "") return value;
  }
  return "";
}


function excelValue(value) {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value) || (typeof value === "object" && !(value instanceof Date))) {
    return JSON.stringify(value);
  }
  if (typeof value === "string" && value.startsWith("=")) return `'${value}`;
  return value;
}


function groupId(group) {
  const matchInfo = group.match_info && typeof group.match_info === "object" ? group.match_info : {};
  return normalizeText(group.uuid || group["故障组ID"] || matchInfo.uuid || "");
}


function alarmRecords(group) {
  const validItems = (items) =>
    (Array.isArray(items) ? items : []).filter(
      (item) => item && typeof item === "object" && !Array.isArray(item),
    );

  let alarms = validItems(group.alarms);
  if (alarms.length) return alarms.map((alarm) => ({ alarm, fallbackNeId: "" }));

  alarms = validItems(group.symptoms);
  if (alarms.length) return alarms.map((alarm) => ({ alarm, fallbackNeId: "" }));

  const matchInfo = group.match_info && typeof group.match_info === "object" ? group.match_info : {};
  alarms = validItems(matchInfo.symptoms);
  if (alarms.length) return alarms.map((alarm) => ({ alarm, fallbackNeId: "" }));

  const result = [];
  const neInfo = group.ne_info && typeof group.ne_info === "object" ? group.ne_info : {};
  for (const [neId, info] of Object.entries(neInfo)) {
    if (!info || typeof info !== "object") continue;
    for (const alarm of validItems(info.alarm)) {
      result.push({ alarm, fallbackNeId: normalizeText(neId) });
    }
  }
  return result;
}


function deviceContext(group, alarm, fallbackNeId) {
  const neId = normalizeText(firstValue(alarm, FIELD_ALIASES["告警源"])) || fallbackNeId;
  const neInfo = group.ne_info && typeof group.ne_info === "object" ? group.ne_info : {};
  const info = neId && neInfo[neId] && typeof neInfo[neId] === "object" ? neInfo[neId] : {};
  return { neId, info };
}


export function groupAlarmRows(group) {
  const rows = [];
  const uuid = groupId(group);
  for (const [offset, item] of alarmRecords(group).entries()) {
    const { alarm, fallbackNeId } = item;
    const { neId, info } = deviceContext(group, alarm, fallbackNeId);
    const row = Object.fromEntries(MAIN_COLUMNS.map((column) => [column, ""]));
    row["故障组ID"] = uuid;
    row["告警序号"] = offset + 1;

    for (const [column, aliases] of Object.entries(FIELD_ALIASES)) {
      row[column] = excelValue(firstValue(alarm, aliases));
    }
    row["告警源"] ||= neId;
    row["设备名称"] ||= excelValue(info.name);
    row["设备域"] ||= excelValue(info.domain);
    row["站点ID"] ||= excelValue(info.site_id);
    row["站点名称"] ||= excelValue(info.site_name);
    row["站点类型"] ||= excelValue(info.site_type);
    row["区域ID"] ||= excelValue(info.region_id);
    row["设备厂家"] ||= excelValue(info.manufacturer);
    row["经度"] ||= excelValue(info.longitude);
    row["纬度"] ||= excelValue(info.latitude);

    for (const [rawKey, rawValue] of Object.entries(alarm)) {
      const key = normalizeText(rawKey);
      if (key && !CONSUMED_ALARM_FIELDS.has(key) && !(key in row)) {
        row[key] = excelValue(rawValue);
      }
    }
    row.__originalOrder = offset;
    rows.push(row);
  }
  return rows;
}


function sortedRows(rows) {
  const collator = new Intl.Collator("zh-CN", { numeric: true, sensitivity: "base" });
  return [...rows].sort((left, right) => {
    for (const field of ["站点ID", "设备厂家", "告警源"]) {
      const compared = collator.compare(normalizeText(left[field]), normalizeText(right[field]));
      if (compared) return compared;
    }
    return left.__originalOrder - right.__originalOrder;
  });
}


function columnName(index) {
  let number = index + 1;
  let name = "";
  while (number > 0) {
    const remainder = (number - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    number = Math.floor((number - 1) / 26);
  }
  return name;
}


function hierarchySpans(rows, fields) {
  const spans = [];
  for (let level = 0; level < fields.length; level += 1) {
    const field = fields[level];
    let start = 0;
    while (start < rows.length) {
      let end = start;
      while (
        end + 1 < rows.length &&
        fields.slice(0, level + 1).every(
          (parentField) => normalizeText(rows[end + 1][parentField]) === normalizeText(rows[start][parentField]),
        )
      ) {
        end += 1;
      }
      spans.push({ level, field, start, end });
      start = end + 1;
    }
  }
  return spans;
}


async function readGroups(inputPath) {
  const text = (await fs.readFile(inputPath, "utf8")).replace(/^\uFEFF/, "");
  const groups = [];
  for (const [offset, rawLine] of text.split(/\r?\n/).entries()) {
    const line = rawLine.trim();
    if (!line) continue;
    let group;
    try {
      group = JSON.parse(line);
    } catch (error) {
      throw new Error(`${inputPath} 第 ${offset + 1} 行 JSON 解析失败: ${error.message}`);
    }
    if (!group || typeof group !== "object" || Array.isArray(group)) {
      throw new Error(`${inputPath} 第 ${offset + 1} 行必须是 JSON 对象`);
    }
    groups.push(group);
  }
  return groups;
}


export async function buildAlarmWorkbook(inputPath) {
  const { Workbook } = await loadArtifactTool();
  const groups = await readGroups(inputPath);
  const rows = sortedRows(groups.flatMap(groupAlarmRows));
  const extraColumns = [];
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (key !== "__originalOrder" && !MAIN_COLUMNS.includes(key) && !extraColumns.includes(key)) {
        extraColumns.push(key);
      }
    }
  }
  const columns = [...MAIN_COLUMNS, ...extraColumns];

  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("告警明细");
  sheet.showGridLines = false;
  const matrix = [columns, ...rows.map((row) => columns.map((column) => excelValue(row[column])))];
  const lastColumn = columnName(columns.length - 1);
  sheet.getRange(`A1:${lastColumn}${matrix.length}`).values = matrix;

  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format = {
    fill: "#145A6A",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "medium", color: "#0B3C49" },
  };
  header.format.rowHeight = 30;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(3);

  if (rows.length) {
    const body = sheet.getRange(`A2:${lastColumn}${rows.length + 1}`);
    body.format = {
      verticalAlignment: "center",
      wrapText: true,
      borders: {
        insideHorizontal: { style: "thin", color: "#D9E2E7" },
        bottom: { style: "thin", color: "#AABBC3" },
      },
    };
    body.format.rowHeight = 24;

    const hierarchyFields = ["站点ID", "设备厂家", "告警源"];
    const hierarchyColors = ["#E8F2F5", "#EEF5E8", "#FFF5DA"];
    for (const span of hierarchySpans(rows, hierarchyFields)) {
      const excelColumn = columnName(span.level);
      const startRow = span.start + 2;
      const endRow = span.end + 2;
      const range = sheet.getRange(`${excelColumn}${startRow}:${excelColumn}${endRow}`);
      if (endRow > startRow) range.merge();
      range.format = {
        fill: hierarchyColors[span.level],
        font: { bold: span.level < 2, color: "#24363D" },
        horizontalAlignment: "center",
        verticalAlignment: "center",
        wrapText: true,
        borders: { preset: "outside", style: "thin", color: "#AABBC3" },
      };
    }
  }

  const widths = [18, 15, 24, 38, 21, 21, 21];
  for (let index = 0; index < columns.length; index += 1) {
    const range = sheet.getRange(`${columnName(index)}1:${columnName(index)}${matrix.length}`);
    range.format.columnWidth = widths[index] || 17;
  }
  return { workbook, alarmCount: rows.length, columns };
}


export async function exportJsonlFile(inputPath, outputPath) {
  const { SpreadsheetFile } = await loadArtifactTool();
  const { workbook, alarmCount } = await buildAlarmWorkbook(inputPath);
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  return alarmCount;
}


export async function exportPerFileDirectory(inputDir, outputDir = inputDir) {
  const entries = await fs.readdir(inputDir, { withFileTypes: true });
  const inputFiles = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".jsonl"))
    .map((entry) => entry.name)
    .sort();
  const stats = {
    input_dir: inputDir,
    output_dir: outputDir,
    input_file_count: inputFiles.length,
    output_file_count: 0,
    alarm_count: 0,
  };
  for (const filename of inputFiles) {
    const outputName = `${path.parse(filename).name}.xlsx`;
    stats.alarm_count += await exportJsonlFile(
      path.join(inputDir, filename),
      path.join(outputDir, outputName),
    );
    stats.output_file_count += 1;
  }
  return stats;
}


function printUsage() {
  console.log(
    "用法: node microwave_topic/export_per_file_alarm_xlsx.mjs <JSONL目录|单个JSONL> [XLSX输出目录|单个XLSX]",
  );
}


async function main() {
  const args = process.argv.slice(2);
  if (!args.length || args.includes("-h") || args.includes("--help")) {
    printUsage();
    return;
  }
  const inputPath = path.resolve(args[0]);
  const inputStats = await fs.stat(inputPath).catch(() => null);
  if (!inputStats) throw new Error(`输入不存在: ${inputPath}`);

  let stats;
  if (inputStats.isDirectory()) {
    const outputDir = path.resolve(args[1] || inputPath);
    stats = await exportPerFileDirectory(inputPath, outputDir);
  } else {
    const outputPath = path.resolve(args[1] || inputPath.replace(/\.jsonl$/i, ".xlsx"));
    const alarmCount = await exportJsonlFile(inputPath, outputPath);
    stats = { input: inputPath, output: outputPath, output_file_count: 1, alarm_count: alarmCount };
  }
  console.log(JSON.stringify(stats, null, 2));
}


const isMain = process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (isMain) {
  main().catch((error) => {
    console.error(`错误: ${error.message}`);
    process.exitCode = 1;
  });
}
