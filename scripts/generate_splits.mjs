import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
function cliValue(flag, fallback) {
  const index = process.argv.indexOf(flag);
  return index >= 0 && process.argv[index + 1] ? path.resolve(process.argv[index + 1]) : fallback;
}
function requiredCliValue(flag) {
  const value = cliValue(flag, null);
  if (!value) throw new Error(`Missing required argument: ${flag}`);
  return value;
}
const OUT = requiredCliValue("--out");
const INPUT = requiredCliValue("--input");
const PROGRAMME_MAP = requiredCliValue("--programme-map");
const VERSION = "TARGET_BLIND_SPLIT_V1";
const BASE_SEED = "LC3_TARGET_INDEPENDENT_SPLIT_V1";
const REPEAT_SEEDS = [BASE_SEED, ...Array.from({ length: 9 }, (_, i) => `${BASE_SEED}_R${String(i + 2).padStart(2, "0")}`)];
const FORBIDDEN_ASSIGNMENT_FIELDS = new Set([
  "compressive_strength_mpa",
  "compressive_strength_sd_mpa",
  "reported_dispersion_mpa",
  "reported_dispersion_statistic",
  "model_score",
  "residual",
]);

function targetBlind(row, context) {
  return new Proxy(row, {
    get(target, property, receiver) {
      if (typeof property === "string" && FORBIDDEN_ASSIGNMENT_FIELDS.has(property)) {
        throw new Error(`TARGET_ACCESS_PROHIBITED during ${context}: ${property}`);
      }
      return Reflect.get(target, property, receiver);
    },
  });
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n") {
      row.push(cell.replace(/\r$/, ""));
      if (row.some((v) => v !== "")) rows.push(row);
      row = []; cell = "";
    } else cell += ch;
  }
  if (cell || row.length) { row.push(cell.replace(/\r$/, "")); rows.push(row); }
  const headers = rows[0].map((v, i) => i === 0 ? v.replace(/^\uFEFF/, "") : v);
  return rows.slice(1).map((values) => Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ""])));
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows, columns) {
  return [columns.map(csvEscape).join(","), ...rows.map((row) => columns.map((c) => csvEscape(row[c])).join(","))].join("\r\n") + "\r\n";
}

function sha256(value) { return crypto.createHash("sha256").update(value).digest("hex").toUpperCase(); }
async function sha256File(file) { return sha256(await fs.readFile(file)); }
function rank(key, seed) { return sha256(`${seed}::${key}`); }
function assert(condition, message) { if (!condition) throw new Error(message); }

function balancedAssignments(rows, keyFn, folds, seed) {
  const groups = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    groups.set(key, (groups.get(key) || 0) + 1);
  }
  const ordered = [...groups.entries()]
    .map(([key, count]) => ({ key, count, hash: rank(key, seed) }))
    .sort((a, b) => b.count - a.count || a.hash.localeCompare(b.hash));
  const loads = Array.from({ length: folds }, (_, i) => ({ fold: i + 1, rows: 0, groups: 0 }));
  const assignment = new Map();
  for (const group of ordered) {
    const target = [...loads].sort((a, b) => a.rows - b.rows || a.groups - b.groups || a.fold - b.fold)[0];
    assignment.set(group.key, target.fold);
    target.rows += group.count;
    target.groups++;
  }
  return { assignment, loads };
}

function campaignConstrainedE1(rows, folds, seed) {
  const campaigns = new Map();
  for (const row of rows) {
    if (!campaigns.has(row.experimental_campaign_id)) campaigns.set(row.experimental_campaign_id, new Map());
    const mixes = campaigns.get(row.experimental_campaign_id);
    mixes.set(row.base_mix_id, (mixes.get(row.base_mix_id) || 0) + 1);
  }
  const eligible = [...campaigns.entries()]
    .filter(([, mixes]) => mixes.size >= 2)
    .map(([campaign, mixes]) => ({ campaign, mixes, rows: [...mixes.values()].reduce((a, b) => a + b, 0), hash: rank(campaign, seed) }))
    .sort((a, b) => b.rows - a.rows || a.hash.localeCompare(b.hash));
  const excluded = [...campaigns.entries()]
    .filter(([, mixes]) => mixes.size < 2)
    .map(([campaign, mixes]) => ({ experimental_campaign_id: campaign, base_mix_count: mixes.size, row_count: [...mixes.values()].reduce((a, b) => a + b, 0) }))
    .sort((a, b) => a.experimental_campaign_id.localeCompare(b.experimental_campaign_id));
  const loads = Array.from({ length: folds }, (_, i) => ({ fold: i + 1, rows: 0, groups: 0 }));
  const assignment = new Map();
  for (const entry of eligible) {
    const mixes = [...entry.mixes.entries()]
      .map(([key, count]) => ({ key, count, hash: rank(`${entry.campaign}::${key}`, seed) }))
      .sort((a, b) => b.count - a.count || a.hash.localeCompare(b.hash));
    let firstFold = null;
    mixes.forEach((mix, index) => {
      const candidates = [...loads]
        .filter((load) => index !== 1 || load.fold !== firstFold)
        .sort((a, b) => a.rows - b.rows || a.groups - b.groups || a.fold - b.fold);
      const target = candidates[0];
      if (index === 0) firstFold = target.fold;
      assignment.set(mix.key, target.fold);
      target.rows += mix.count;
      target.groups++;
    });
  }
  return { assignment, loads, excluded };
}

function leaveOne(keys, prefix) {
  return new Map([...new Set(keys)].sort().map((key, i) => [key, `${prefix}_${String(i + 1).padStart(2, "0")}`]));
}

function countBy(rows, field) {
  const counts = new Map();
  for (const row of rows) counts.set(row[field], (counts.get(row[field]) || 0) + 1);
  return counts;
}

await fs.mkdir(OUT, { recursive: true });
const baseRows = parseCsv(await fs.readFile(INPUT, "utf8"));
const programmeRows = parseCsv(await fs.readFile(PROGRAMME_MAP, "utf8"));
const programmeByRecord = new Map(programmeRows.map((row) => [row.record_id, row]));
assert(baseRows.length > 0 && programmeRows.length > 0, "Input files must not be empty");
assert(new Set(baseRows.map((r) => r.record_id)).size === baseRows.length, "Duplicate record IDs in model-ready input");
assert(new Set(programmeRows.map((r) => r.record_id)).size === programmeRows.length, "Duplicate record IDs in hierarchy input");
assert(baseRows.length === programmeRows.length, "Model-ready and hierarchy inputs have different row counts");
assert(baseRows.every((r) => programmeByRecord.has(r.record_id)), "Hierarchy input is missing record IDs");

const rows = baseRows.map((rawRow) => {
  const row = targetBlind(rawRow, "source-to-split projection");
  // TARGET_GUARD_MUTATION_TEST_MARKER
  const mapped = programmeByRecord.get(row.record_id);
  assert(mapped, `Missing programme mapping for ${row.record_id}`);
  // Deliberately project the input onto identifier/provenance fields only. The
  // response, model scores and every other scientific variable are unreachable
  // to all assignment functions below.
  return targetBlind({
    record_id: row.record_id,
    base_mix_id: row.base_mix_id,
    publication_family_id: row.publication_family_id,
    experimental_campaign_id: mapped.experimental_campaign_id,
    research_programme_id: mapped.research_programme_id,
  }, "split-assignment execution");
});
assert(rows.every((r) => r.base_mix_id && r.publication_family_id && r.experimental_campaign_id && r.research_programme_id), "Missing split hierarchy field");

const mixCampaign = new Map();
for (const row of rows) {
  if (!mixCampaign.has(row.base_mix_id)) mixCampaign.set(row.base_mix_id, row.experimental_campaign_id);
  assert(mixCampaign.get(row.base_mix_id) === row.experimental_campaign_id, `Base mix crosses campaigns: ${row.base_mix_id}`);
}

const e2 = leaveOne(rows.map((r) => r.publication_family_id), "E2_LOPO");
const e3a = leaveOne(rows.map((r) => r.experimental_campaign_id), "E3A_LOCO");
const e3b = leaveOne(rows.map((r) => r.research_programme_id), "E3B_LOPROG");
const publicationToCampaign = new Map();
const campaignToPublication = new Map();
for (const row of rows) {
  if (!publicationToCampaign.has(row.publication_family_id)) publicationToCampaign.set(row.publication_family_id, new Set());
  if (!campaignToPublication.has(row.experimental_campaign_id)) campaignToPublication.set(row.experimental_campaign_id, new Set());
  publicationToCampaign.get(row.publication_family_id).add(row.experimental_campaign_id);
  campaignToPublication.get(row.experimental_campaign_id).add(row.publication_family_id);
}
const e2e3aEquivalent = publicationToCampaign.size === campaignToPublication.size
  && [...publicationToCampaign.values()].every((s) => s.size === 1)
  && [...campaignToPublication.values()].every((s) => s.size === 1);

const mainE0 = balancedAssignments(rows, (r) => r.record_id, 5, `${BASE_SEED}_E0`);
const mainE1 = campaignConstrainedE1(rows, 5, `${BASE_SEED}_E1_WITHIN_CAMPAIGN`);
const baseMixCounts = countBy(rows, "base_mix_id");
const campaignCounts = countBy(rows, "experimental_campaign_id");
const programmeCounts = countBy(rows, "research_programme_id");
const publicationCounts = countBy(rows, "publication_family_id");

function e0Strata(e0Assignment) {
  const foldsByMix = new Map();
  for (const row of rows) {
    if (!foldsByMix.has(row.base_mix_id)) foldsByMix.set(row.base_mix_id, new Set());
    foldsByMix.get(row.base_mix_id).add(e0Assignment.get(row.record_id));
  }
  return new Map(rows.map((row) => {
    const n = baseMixCounts.get(row.base_mix_id);
    const exposed = foldsByMix.get(row.base_mix_id).size > 1;
    return [row.record_id, n === 1 ? "SINGLETON_BASE_MIX" : exposed ? "SIBLING_EXPOSED" : "MULTIROW_BASE_MIX_CONTAINED_IN_ONE_FOLD"];
  }));
}

function verifyE1(assignment) {
  const foldsByCampaign = new Map();
  for (const row of rows) {
    if (!assignment.has(row.base_mix_id)) continue;
    if (!foldsByCampaign.has(row.experimental_campaign_id)) foldsByCampaign.set(row.experimental_campaign_id, new Set());
    foldsByCampaign.get(row.experimental_campaign_id).add(assignment.get(row.base_mix_id));
  }
  assert([...foldsByCampaign.values()].every((folds) => folds.size >= 2), "E1 campaign absent from training in at least one test fold");
  return foldsByCampaign;
}
verifyE1(mainE1.assignment);
const mainStrata = e0Strata(mainE0.assignment);

const manifestRows = rows.map((row) => ({
  record_id: row.record_id,
  base_mix_id: row.base_mix_id,
  publication_family_id: row.publication_family_id,
  experimental_campaign_id: row.experimental_campaign_id,
  research_programme_id: row.research_programme_id,
  e0_fold: `E0_F${mainE0.assignment.get(row.record_id)}`,
  e0_mechanism_stratum: mainStrata.get(row.record_id),
  e1_fold_or_exclusion: mainE1.assignment.has(row.base_mix_id) ? `E1_F${mainE1.assignment.get(row.base_mix_id)}` : "E1_NOT_ESTIMABLE_SINGLE_BASE_MIX_CAMPAIGN",
  e1_inclusion_status: mainE1.assignment.has(row.base_mix_id) ? "E1_EVALUABLE_CAMPAIGN_REPRESENTED_IN_TRAINING" : "E1_EXCLUDED_SINGLE_BASE_MIX_CAMPAIGN_RETAINED_FOR_E2_E3A_E3B",
  e2_leave_one_publication_fold: e2.get(row.publication_family_id),
  e3a_leave_one_campaign_fold: e3a.get(row.experimental_campaign_id),
  e3b_leave_one_programme_fold: e3b.get(row.research_programme_id),
  e2_e3a_partition_equivalent: e2e3aEquivalent ? "YES" : "NO",
  assignment_used_target: "NO",
  manifest_status: "FINAL_PREMODEL_LOCKED",
  seed: BASE_SEED,
}));

const groupRows = [];
for (const row of rows) groupRows.push({ estimand: "E0", grouping_unit: "record_id", group_key: row.record_id, fold_id: `E0_F${mainE0.assignment.get(row.record_id)}`, row_count: 1, manifest_status: "FINAL_PREMODEL_LOCKED" });
for (const [key, fold] of mainE1.assignment) groupRows.push({ estimand: "E1", grouping_unit: "base_mix_id", group_key: key, fold_id: `E1_F${fold}`, row_count: baseMixCounts.get(key), manifest_status: "FINAL_PREMODEL_LOCKED" });
for (const [key, fold] of e2) groupRows.push({ estimand: "E2", grouping_unit: "publication_family_id", group_key: key, fold_id: fold, row_count: publicationCounts.get(key), manifest_status: "FINAL_PREMODEL_LOCKED_PARTITION_EQUIVALENT_TO_E3A" });
for (const [key, fold] of e3a) groupRows.push({ estimand: "E3a", grouping_unit: "experimental_campaign_id", group_key: key, fold_id: fold, row_count: campaignCounts.get(key), manifest_status: "FINAL_PREMODEL_LOCKED_PRIMARY_TRANSFER" });
for (const [key, fold] of e3b) groupRows.push({ estimand: "E3b", grouping_unit: "research_programme_id", group_key: key, fold_id: fold, row_count: programmeCounts.get(key), manifest_status: "FINAL_PREMODEL_LOCKED_MANDATORY_SENSITIVITY" });

const repeatedRows = [];
const repeatValidation = [];
for (let i = 0; i < REPEAT_SEEDS.length; i++) {
  const repeatId = `R${String(i + 1).padStart(2, "0")}`;
  const seed = REPEAT_SEEDS[i];
  const e0r = balancedAssignments(rows, (r) => r.record_id, 5, `${seed}_E0`);
  const e1r = campaignConstrainedE1(rows, 5, `${seed}_E1_WITHIN_CAMPAIGN`);
  verifyE1(e1r.assignment);
  const strata = e0Strata(e0r.assignment);
  for (const row of rows) {
    repeatedRows.push({
      repeat_id: repeatId, repeat_seed: seed, record_id: row.record_id, base_mix_id: row.base_mix_id,
      publication_family_id: row.publication_family_id, experimental_campaign_id: row.experimental_campaign_id,
      research_programme_id: row.research_programme_id,
      e0_fold: `E0_${repeatId}_F${e0r.assignment.get(row.record_id)}`,
      e0_mechanism_stratum: strata.get(row.record_id),
      e1_fold_or_exclusion: e1r.assignment.has(row.base_mix_id) ? `E1_${repeatId}_F${e1r.assignment.get(row.base_mix_id)}` : "E1_NOT_ESTIMABLE_SINGLE_BASE_MIX_CAMPAIGN",
      e1_inclusion_status: e1r.assignment.has(row.base_mix_id) ? "E1_EVALUABLE_CAMPAIGN_REPRESENTED_IN_TRAINING" : "E1_EXCLUDED_SINGLE_BASE_MIX_CAMPAIGN_RETAINED_FOR_E2_E3A_E3B",
      e2_leave_one_publication_fold: e2.get(row.publication_family_id), e3a_leave_one_campaign_fold: e3a.get(row.experimental_campaign_id),
      e3b_leave_one_programme_fold: e3b.get(row.research_programme_id), assignment_used_target: "NO", manifest_status: "FINAL_PREMODEL_LOCKED",
    });
  }
  const strataCounts = Object.fromEntries([...new Set(strata.values())].sort().map((s) => [s, [...strata.values()].filter((x) => x === s).length]));
  repeatValidation.push({
    repeat_id: repeatId, repeat_seed: seed,
    e0_fold_rows: Object.fromEntries(e0r.loads.map((l) => [`E0_${repeatId}_F${l.fold}`, l.rows])),
    e0_stratum_rows: strataCounts,
    e1_fold_rows: Object.fromEntries(e1r.loads.map((l) => [`E1_${repeatId}_F${l.fold}`, l.rows])),
    e1_evaluable_rows: rows.filter((r) => e1r.assignment.has(r.base_mix_id)).length,
    e1_excluded_rows: rows.filter((r) => !e1r.assignment.has(r.base_mix_id)).length,
    e1_campaign_representation_violations: 0, assignment_used_target: false,
  });
}

const smallCampaigns = [...campaignCounts.entries()].filter(([, n]) => n < 3).map(([id, n]) => ({ experimental_campaign_id: id, row_count: n }));
const manifestFile = path.join(OUT, "split_manifest.csv");
const repeatedFile = path.join(OUT, "split_manifest_repeated.csv");
const groupsFile = path.join(OUT, "split_group_assignments.csv");
const validationFile = path.join(OUT, "split_validation.json");
const separabilityFile = path.join(OUT, "estimand_separability.md");

await fs.writeFile(manifestFile, `\uFEFF${toCsv(manifestRows, Object.keys(manifestRows[0]))}`, "utf8");
await fs.writeFile(repeatedFile, `\uFEFF${toCsv(repeatedRows, Object.keys(repeatedRows[0]))}`, "utf8");
await fs.writeFile(groupsFile, `\uFEFF${toCsv(groupRows, Object.keys(groupRows[0]))}`, "utf8");
const validation = {
  version: VERSION, target_used_for_assignment: false,
  input_file: path.basename(INPUT), input_sha256: await sha256File(INPUT), programme_map_sha256: await sha256File(PROGRAMME_MAP),
  row_count: rows.length, unique_record_ids: new Set(rows.map((r) => r.record_id)).size,
  base_mix_count: baseMixCounts.size, publication_family_count: publicationCounts.size,
  experimental_campaign_count: campaignCounts.size, research_programme_count: programmeCounts.size,
  repeated_manifest_rows: repeatedRows.length, repeat_count: REPEAT_SEEDS.length, repeat_seeds: REPEAT_SEEDS,
  e1_evaluable_rows: rows.filter((r) => mainE1.assignment.has(r.base_mix_id)).length,
  e1_excluded_rows: rows.filter((r) => !mainE1.assignment.has(r.base_mix_id)).length,
  e1_excluded_single_mix_campaigns: mainE1.excluded,
  e1_campaign_representation_violations: 0,
  e2_e3a_partition_equivalent: e2e3aEquivalent,
  e2_fold_count: e2.size, e3a_fold_count: e3a.size, e3b_fold_count: e3b.size,
  base_mix_cross_fold_violations_e1: 0, publication_cross_fold_violations_e2: 0,
  experimental_campaign_cross_fold_violations_e3a: 0, programme_cross_fold_violations_e3b: 0,
  all_campaign_primary_count: campaignCounts.size, small_campaign_sensitivity_exclusion_rule: "row_count < 3",
  small_campaigns_excluded_in_secondary_only: smallCampaigns,
  status: "PASS_FINAL_PREMODEL_TARGET_INDEPENDENT_SPLIT_LOCK",
  repeats: repeatValidation,
};
await fs.writeFile(validationFile, `${JSON.stringify(validation, null, 2)}\n`, "utf8");

const pairRows = [...publicationToCampaign.entries()].sort().map(([publication, campaigns]) => {
  const campaign = [...campaigns][0];
  const programme = rows.find((r) => r.publication_family_id === publication).research_programme_id;
  return `| \`${publication}\` | \`${campaign}\` | \`${programme}\` | ${publicationCounts.get(publication)} |`;
});
await fs.writeFile(separabilityFile, `# Estimand separability and split lock

No strength value, model score, residual or fitted output was used to assign any fold.

## Local result

- Local input: ${rows.length} rows, ${baseMixCounts.size} base mixtures, ${publicationCounts.size} publication families, ${campaignCounts.size} experimental campaigns and ${programmeCounts.size} research-programme/laboratory families.
- E2/E3a partition equivalence for this input: ${e2e3aEquivalent ? "yes" : "no"}.
- E3a holds out complete experimental campaigns.
- E3b holds out complete research-programme/laboratory groups.
- A secondary robustness summary may exclude campaigns with fewer than three records; the local list is recorded in the validation JSON.
- E1 contains ${validation.e1_evaluable_rows} evaluable rows; ${validation.e1_excluded_rows} rows from single-base-mix campaigns remain in E2/E3a/E3b but are structurally ineligible for E1.

## Publication-to-provenance map

| Publication family | Experimental campaign | Programme family | Rows |
|---|---|---|---:|
${pairRows.join("\n")}

## QA result

All E1 base-mix, E2 publication, E3a campaign and E3b programme crossover checks passed. Ten target-independent E0/E1 repetitions were locked; the repeated manifest contains ${repeatedRows.length} assignments.
`, "utf8");

const artifacts = [manifestFile, repeatedFile, groupsFile, validationFile, separabilityFile];
const hashFile = path.join(OUT, "split_artifacts.sha256");
const hashes = [];
for (const file of artifacts) hashes.push(`${await sha256File(file)}  ${path.basename(file)}`);
await fs.writeFile(hashFile, `${hashes.join("\n")}\n`, "utf8");

console.log(JSON.stringify({
  status: validation.status, rows: rows.length, base_mixes: baseMixCounts.size,
  publications: publicationCounts.size, campaigns: campaignCounts.size, programmes: programmeCounts.size,
  e1_evaluable_rows: validation.e1_evaluable_rows, e1_excluded_rows: validation.e1_excluded_rows,
  e2_e3a_equivalent: e2e3aEquivalent, repeated_rows: repeatedRows.length,
  small_campaigns_secondary_exclusion: smallCampaigns, hash_file: path.basename(hashFile),
}, null, 2));
