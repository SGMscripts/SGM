-- @description SGM Perch and Birdnet Detect
-- @version 1.1
-- @author Sruthin + Codex
-- @about
--   ReaImGui front-end for Perch v2 and BirdNET detection.
--   Keeps the existing non-UI Perch script untouched.

local r = reaper
local SEP = package.config:sub(1, 1)

if not r.ImGui_CreateContext then
  r.MB(
    "This script needs ReaImGui.\nInstall 'ReaImGui: ReaScript binding for Dear ImGui' from ReaPack.",
    "SGM Perch and Birdnet Detect",
    0
  )
  return
end

local SCRIPT_TITLE = "SGM Perch and Birdnet Detect"
local SCRIPTS_DIR = r.GetResourcePath() .. SEP .. "Scripts"
local DEFAULT_BUNDLE_DIR = SCRIPTS_DIR .. SEP .. "SFX_BirdDetect_Pack"
local DEFAULT_HELPER_PATH = SCRIPTS_DIR .. SEP .. "sfx_perch_auto_name_helper.py"
local DEFAULT_EBIRD_UPDATER_PATH = SCRIPTS_DIR .. SEP .. "sfx_ebird_local_list_updater.py"
local EXT_SECTION = "SFX_BirdDetect_IMGUI_Sruthin"
local COMMON_PYTHON_CANDIDATES = {
  "/opt/homebrew/bin/python3.12",
  "/opt/homebrew/bin/python3",
  "/usr/local/bin/python3.12",
  "/usr/local/bin/python3",
  "/usr/bin/python3",
}
local PERCH_MODEL_PATH = ""
local PERCH_LABELS_PATH = ""
local LOCAL_BIRDS_FILE = ""

local REQUEST_FILE = "/tmp/sfx_perch_imgui_request.json"
local RESPONSE_FILE = "/tmp/sfx_perch_imgui_response.tsv"
local LOG_FILE = "/tmp/sfx_perch_auto_name.log"

local state = {
  engine_idx = 0, -- 0 perch_v2, 1 birdnet
  name_threshold_pct = 35.0,
  marker_threshold_pct = 25.0,
  detections_per_segment = 5,
  list_mode_idx = 0, -- 0 local, 1 aves, 2 mammalia, 3 insecta, 4 amphibia, 5 all
  strict_local = true,
  latitude = "",
  longitude = "",
  top_k = 3,
  name_top_n = 3,
  name_as_list = true,
  rename_take = true,
  write_notes = true,
  add_markers = true,
  add_fallback_marker = false,
  marker_gap_s = 3.0,
  onset_snap = true,
  onset_lookback_s = 0.35,
  onset_lookahead_s = 0.90,
  birdnet_max_audio_min = 5.0,
  marker_offset_ms = 0,
  max_take_name = 120,
  bundle_dir_override = "",
  setup_script_override = "",
  helper_path_override = "",
  perch_python_override = "",
  birdnet_python_override = "",
  perch_model_override = "",
  perch_labels_override = "",
  local_birds_override = "",
  ebird_updater_override = "",
  ebird_api_key = "",
  ebird_radius_km = 50,
  ebird_back_days = 30,
  ebird_max_results = 10000,
  status_line = "Ready.",
  last_summary = "",
}

local ENGINE_ITEMS = "Perch v2 (ONNX)\0BirdNET\0\0"
local LIST_MODE_ITEMS = "Local List\0Aves\0Mammalia\0Insecta\0Amphibia\0All Labels\0\0"

local function trim(s)
  return tostring(s or ""):match("^%s*(.-)%s*$") or ""
end

local function get_ext_state(key, default)
  local v = r.GetExtState(EXT_SECTION, key)
  if not v or v == "" then return default or "" end
  return v
end

local function load_path_overrides()
  state.bundle_dir_override = get_ext_state("bundle_dir_override", "")
  state.setup_script_override = get_ext_state("setup_script_override", "")
  state.helper_path_override = get_ext_state("helper_path_override", "")
  state.perch_python_override = get_ext_state("perch_python_override", "")
  state.birdnet_python_override = get_ext_state("birdnet_python_override", "")
  state.perch_model_override = get_ext_state("perch_model_override", "")
  state.perch_labels_override = get_ext_state("perch_labels_override", "")
  state.local_birds_override = get_ext_state("local_birds_override", "")
  state.ebird_updater_override = get_ext_state("ebird_updater_override", "")
  state.ebird_api_key = get_ext_state("ebird_api_key", "")
  state.ebird_radius_km = tonumber(get_ext_state("ebird_radius_km", "50")) or 50
  state.ebird_back_days = tonumber(get_ext_state("ebird_back_days", "30")) or 30
  state.ebird_max_results = tonumber(get_ext_state("ebird_max_results", "10000")) or 10000
end

local function save_path_overrides()
  r.SetExtState(EXT_SECTION, "bundle_dir_override", trim(state.bundle_dir_override), true)
  r.SetExtState(EXT_SECTION, "setup_script_override", trim(state.setup_script_override), true)
  r.SetExtState(EXT_SECTION, "helper_path_override", trim(state.helper_path_override), true)
  r.SetExtState(EXT_SECTION, "perch_python_override", trim(state.perch_python_override), true)
  r.SetExtState(EXT_SECTION, "birdnet_python_override", trim(state.birdnet_python_override), true)
  r.SetExtState(EXT_SECTION, "perch_model_override", trim(state.perch_model_override), true)
  r.SetExtState(EXT_SECTION, "perch_labels_override", trim(state.perch_labels_override), true)
  r.SetExtState(EXT_SECTION, "local_birds_override", trim(state.local_birds_override), true)
  r.SetExtState(EXT_SECTION, "ebird_updater_override", trim(state.ebird_updater_override), true)
  r.SetExtState(EXT_SECTION, "ebird_api_key", trim(state.ebird_api_key), true)
  r.SetExtState(EXT_SECTION, "ebird_radius_km", tostring(math.floor(tonumber(state.ebird_radius_km) or 50)), true)
  r.SetExtState(EXT_SECTION, "ebird_back_days", tostring(math.floor(tonumber(state.ebird_back_days) or 30)), true)
  r.SetExtState(EXT_SECTION, "ebird_max_results", tostring(math.floor(tonumber(state.ebird_max_results) or 10000)), true)
end

local function clear_path_overrides()
  state.bundle_dir_override = ""
  state.setup_script_override = ""
  state.helper_path_override = ""
  state.perch_python_override = ""
  state.birdnet_python_override = ""
  state.perch_model_override = ""
  state.perch_labels_override = ""
  state.local_birds_override = ""
  state.ebird_updater_override = ""
end

load_path_overrides()

local function rgba(rf, gf, bf, af)
  return r.ImGui_ColorConvertDouble4ToU32(rf, gf, bf, af)
end

local UI_COLORS = {
  bg = rgba(0.12, 0.09, 0.07, 0.98),
  bg_alt = rgba(0.18, 0.12, 0.08, 0.98),
  frame = rgba(0.23, 0.18, 0.13, 1.00),
  frame_hover = rgba(0.31, 0.23, 0.16, 1.00),
  frame_active = rgba(0.37, 0.27, 0.19, 1.00),
  button = rgba(0.56, 0.33, 0.11, 1.00),
  button_hover = rgba(0.68, 0.41, 0.14, 1.00),
  button_active = rgba(0.44, 0.26, 0.09, 1.00),
  header = rgba(0.24, 0.31, 0.18, 1.00),
  header_hover = rgba(0.30, 0.40, 0.23, 1.00),
  header_active = rgba(0.38, 0.50, 0.29, 1.00),
  accent_sun = rgba(0.97, 0.78, 0.27, 1.00),
  accent_sky = rgba(0.90, 0.52, 0.25, 1.00),
  accent_leaf = rgba(0.54, 0.72, 0.36, 1.00),
  text_dim = rgba(0.91, 0.84, 0.76, 1.00),
  status_ok = rgba(0.56, 0.84, 0.45, 1.00),
  status_bad = rgba(1.00, 0.38, 0.38, 1.00),
  status_run = rgba(0.97, 0.70, 0.23, 1.00),
  status_neutral = rgba(0.96, 0.90, 0.82, 1.00),
  summary_bg = rgba(0.17, 0.12, 0.09, 0.94),
}

local UI_THEME_COLOR_COUNT = 15
local UI_THEME_VAR_COUNT = 6
local BIRD_FRAMES = {
  "<(' )>",
  "<( ')>",
  "<(  ')>",
  "<( ' )>",
}

local function push_vibrant_theme(ctx)
  r.ImGui_PushStyleVar(ctx, r.ImGui_StyleVar_WindowPadding(), 14, 12)
  r.ImGui_PushStyleVar(ctx, r.ImGui_StyleVar_FramePadding(), 8, 5)
  r.ImGui_PushStyleVar(ctx, r.ImGui_StyleVar_ItemSpacing(), 8, 8)
  r.ImGui_PushStyleVar(ctx, r.ImGui_StyleVar_WindowRounding(), 10)
  r.ImGui_PushStyleVar(ctx, r.ImGui_StyleVar_FrameRounding(), 7)
  r.ImGui_PushStyleVar(ctx, r.ImGui_StyleVar_GrabRounding(), 7)

  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_WindowBg(), UI_COLORS.bg)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_TitleBg(), UI_COLORS.bg_alt)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_TitleBgActive(), UI_COLORS.bg_alt)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_FrameBg(), UI_COLORS.frame)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_FrameBgHovered(), UI_COLORS.frame_hover)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_FrameBgActive(), UI_COLORS.frame_active)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_Button(), UI_COLORS.button)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_ButtonHovered(), UI_COLORS.button_hover)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_ButtonActive(), UI_COLORS.button_active)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_Header(), UI_COLORS.header)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_HeaderHovered(), UI_COLORS.header_hover)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_HeaderActive(), UI_COLORS.header_active)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_CheckMark(), UI_COLORS.accent_sun)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_SliderGrab(), UI_COLORS.accent_sky)
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_SliderGrabActive(), UI_COLORS.accent_leaf)
end

local function pop_vibrant_theme(ctx)
  pcall(r.ImGui_PopStyleColor, ctx, UI_THEME_COLOR_COUNT)
  pcall(r.ImGui_PopStyleVar, ctx, UI_THEME_VAR_COUNT)
end

local function text_colored(ctx, col, text)
  if r.ImGui_TextColored then
    r.ImGui_TextColored(ctx, col, text)
    return
  end
  r.ImGui_PushStyleColor(ctx, r.ImGui_Col_Text(), col)
  r.ImGui_Text(ctx, text)
  pcall(r.ImGui_PopStyleColor, ctx)
end

local function status_color(msg)
  local t = tostring(msg or ""):lower()
  if t:find("error", 1, true) or t:find("not found", 1, true) or t:find("failed", 1, true) then
    return UI_COLORS.status_bad
  end
  if t:find("running", 1, true) then
    return UI_COLORS.status_run
  end
  if t:find("done", 1, true) then
    return UI_COLORS.status_ok
  end
  return UI_COLORS.status_neutral
end

local function animated_bird()
  local idx = (math.floor(os.clock() * 8) % #BIRD_FRAMES) + 1
  local palette = {
    UI_COLORS.accent_leaf,
    UI_COLORS.accent_sun,
    UI_COLORS.accent_sky,
    UI_COLORS.accent_sun,
  }
  return BIRD_FRAMES[idx], palette[idx]
end

local function draw_header(ctx, selected_count)
  text_colored(ctx, UI_COLORS.text_dim, "Selected items: " .. tostring(selected_count))
  r.ImGui_SameLine(ctx)
  if state.engine_idx == 1 then
    text_colored(ctx, UI_COLORS.accent_sky, "Engine: BirdNET")
  else
    text_colored(ctx, UI_COLORS.accent_leaf, "Engine: Perch v2")
  end
end

local function file_exists(path)
  local f = io.open(path, "r")
  if not f then return false end
  f:close()
  return true
end

local function first_existing(paths)
  for _, p in ipairs(paths or {}) do
    if p and p ~= "" and file_exists(p) then
      return p
    end
  end
  return (paths and paths[1]) or ""
end

local function append_paths(dst, src)
  for _, p in ipairs(src or {}) do
    dst[#dst + 1] = p
  end
end

local function active_bundle_dir()
  local p = trim(state.bundle_dir_override)
  if p ~= "" then return p end
  return DEFAULT_BUNDLE_DIR
end

local function active_setup_script()
  local override = trim(state.setup_script_override)
  if override ~= "" then return override end
  local bundle_dir = active_bundle_dir()
  local primary = bundle_dir .. SEP .. "setup_perch_birdnet_macos.sh"
  local legacy = bundle_dir .. SEP .. "setup_macos.sh"
  if file_exists(primary) then return primary end
  if file_exists(legacy) then return legacy end
  return primary
end

local function active_helper_path()
  local override = trim(state.helper_path_override)
  if override ~= "" then return override end
  return DEFAULT_HELPER_PATH
end

local function active_ebird_updater_path()
  local override = trim(state.ebird_updater_override)
  if override ~= "" then return override end
  return DEFAULT_EBIRD_UPDATER_PATH
end

local function perch_python_candidates()
  local bundle_dir = active_bundle_dir()
  local out = {}
  local override = trim(state.perch_python_override)
  if override ~= "" then out[#out + 1] = override end
  append_paths(out, {
    bundle_dir .. SEP .. "venv" .. SEP .. "perch_py312" .. SEP .. "bin" .. SEP .. "python3.12",
    bundle_dir .. SEP .. "venv" .. SEP .. "perch_py312" .. SEP .. "bin" .. SEP .. "python3",
    SCRIPTS_DIR .. SEP .. "sfx_clap_env" .. SEP .. "bin" .. SEP .. "python3.12",
    SCRIPTS_DIR .. SEP .. "sfx_clap_env" .. SEP .. "bin" .. SEP .. "python3",
  })
  append_paths(out, COMMON_PYTHON_CANDIDATES)
  return out
end

local function birdnet_python_candidates()
  local bundle_dir = active_bundle_dir()
  local out = {}
  local override = trim(state.birdnet_python_override)
  if override ~= "" then out[#out + 1] = override end
  append_paths(out, {
    bundle_dir .. SEP .. "venv" .. SEP .. "birdnet_arm64" .. SEP .. "bin" .. SEP .. "python3",
    SCRIPTS_DIR .. SEP .. "sfx_birdnet_arm64_env" .. SEP .. "bin" .. SEP .. "python3",
  })
  append_paths(out, COMMON_PYTHON_CANDIDATES)
  return out
end

local function updater_python_candidates()
  local out = {}
  append_paths(out, birdnet_python_candidates())
  append_paths(out, perch_python_candidates())
  append_paths(out, COMMON_PYTHON_CANDIDATES)
  return out
end

local function resolve_runtime_paths()
  local bundle_dir = active_bundle_dir()

  local perch_model_candidates = {}
  local perch_model_override = trim(state.perch_model_override)
  if perch_model_override ~= "" then perch_model_candidates[#perch_model_candidates + 1] = perch_model_override end
  append_paths(perch_model_candidates, {
    bundle_dir .. SEP .. "models" .. SEP .. "perch_v2" .. SEP .. "perch_v2.onnx",
    SCRIPTS_DIR .. SEP .. "perch_v2.onnx",
  })

  local perch_labels_candidates = {}
  local perch_labels_override = trim(state.perch_labels_override)
  if perch_labels_override ~= "" then perch_labels_candidates[#perch_labels_candidates + 1] = perch_labels_override end
  append_paths(perch_labels_candidates, {
    bundle_dir .. SEP .. "models" .. SEP .. "perch_v2" .. SEP .. "labels.txt",
    SCRIPTS_DIR .. SEP .. "labels.txt",
  })

  local local_birds_candidates = {}
  local local_birds_override = trim(state.local_birds_override)
  if local_birds_override ~= "" then local_birds_candidates[#local_birds_candidates + 1] = local_birds_override end
  append_paths(local_birds_candidates, {
    bundle_dir .. SEP .. "species_list.txt",
    SCRIPTS_DIR .. SEP .. "species_list.txt",
    SCRIPTS_DIR .. SEP .. "local_birds.txt",
  })

  PERCH_MODEL_PATH = first_existing(perch_model_candidates)
  PERCH_LABELS_PATH = first_existing(perch_labels_candidates)
  LOCAL_BIRDS_FILE = first_existing(local_birds_candidates)
end

local function shell_quote(s)
  s = tostring(s or "")
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

local function setup_command_hint()
  local setup_script = active_setup_script()
  if file_exists(setup_script) then
    return "bash " .. shell_quote(setup_script)
  end
  return "(missing setup script: " .. setup_script .. ")"
end

local function launch_setup_script()
  local setup_script = active_setup_script()
  if not file_exists(setup_script) then
    state.status_line = "Setup script not found."
    state.last_summary = "Expected setup script:\n" .. setup_script
    return
  end

  local cmd_parts = {}
  cmd_parts[#cmd_parts + 1] = "SCRIPTS_DIR=" .. shell_quote(SCRIPTS_DIR)

  local perch_model = trim(state.perch_model_override)
  if perch_model ~= "" then
    cmd_parts[#cmd_parts + 1] = "PERCH_MODEL_SRC=" .. shell_quote(perch_model)
  end

  local perch_labels = trim(state.perch_labels_override)
  if perch_labels ~= "" then
    cmd_parts[#cmd_parts + 1] = "PERCH_LABELS_SRC=" .. shell_quote(perch_labels)
  end

  local species_list = trim(state.local_birds_override)
  if species_list ~= "" then
    cmd_parts[#cmd_parts + 1] = "SPECIES_LIST_SRC=" .. shell_quote(species_list)
  end

  cmd_parts[#cmd_parts + 1] = "bash " .. shell_quote(setup_script)
  local cmd = table.concat(cmd_parts, " ")
    .. " >" .. shell_quote(LOG_FILE) .. " 2>&1 &"

  local ok = os.execute(cmd)
  if ok == false then
    state.status_line = "Setup launch failed."
    state.last_summary = "Failed command:\n" .. cmd
    return
  end

  state.status_line = "Setup started in background."
  state.last_summary = "Running setup script:\n"
    .. setup_script
    .. "\n\nLog:\n"
    .. LOG_FILE
end

local function json_escape(s)
  s = tostring(s or "")
  s = s:gsub("\\", "\\\\")
       :gsub('"', '\\"')
       :gsub("\n", "\\n")
       :gsub("\r", "\\r")
       :gsub("\t", "\\t")
  return s
end

local function split_tab(line)
  local out = {}
  for field in (line .. "\t"):gmatch("(.-)\t") do
    out[#out + 1] = field
  end
  return out
end

local function basename(path)
  return (path and path:match("([^/\\]+)$")) or (path or "")
end

local function sanitize_marker_text(s)
  s = tostring(s or "")
  s = s:gsub("[%c]", " ")
  s = s:gsub("%s+", " ")
  s = s:gsub("^%s+", ""):gsub("%s+$", "")
  if #s > 72 then s = s:sub(1, 72) end
  return s
end

local function clamp_name(s, limit)
  s = tostring(s or "")
  if #s <= limit then return s end
  return s:sub(1, math.max(1, limit - 3)) .. "..."
end

local function parse_scene_markers(raw)
  local out = {}
  raw = tostring(raw or "")
  if raw == "" then return out end
  for part in (raw .. "||"):gmatch("(.-)||") do
    local ts, label = part:match("^([%-%d%.]+)@@(.+)$")
    local t = tonumber(ts)
    if t and label and label ~= "" then
      out[#out + 1] = { t = t, label = sanitize_marker_text(label) }
    end
  end
  return out
end

local function read_tsv_results(path)
  local f = io.open(path, "r")
  if not f then return nil end
  local map = {}
  for line in f:lines() do
    if line ~= "" then
      local p = split_tab(line)
      local src_path = p[1] or ""
      if src_path ~= "" then
        map[src_path] = {
          name = p[2] or "",
          action = p[3] or "",
          subcat = p[4] or "",
          material = p[5] or "",
          confidence = tonumber(p[6]) or 0,
          top1 = p[7] or "",
          top2 = p[8] or "",
          top3 = p[9] or "",
          top1_desc = p[10] or "",
          top2_desc = p[11] or "",
          top3_desc = p[12] or "",
          content_type = p[13] or "",
          vehicle_model = p[14] or "",
          scene_markers = p[15] or "",
          list_mode_used = p[16] or "",
          location_used = p[17] or "",
          threshold_mode = p[18] or "",
          name_raw_cutoff = tonumber(p[19]) or 0,
          marker_threshold_mode = p[20] or "",
          marker_raw_cutoff = tonumber(p[21]) or 0,
          raw_top1 = tonumber(p[22]) or 0,
          name_threshold_rel = tonumber(p[23]),
          marker_threshold_rel = tonumber(p[24]),
        }
      end
    end
  end
  f:close()
  return map
end

local function collect_selected_entries()
  local sel_count = r.CountSelectedMediaItems(0)
  local entries = {}
  local by_path = {}
  local unique_paths = {}

  for i = 0, sel_count - 1 do
    local item = r.GetSelectedMediaItem(0, i)
    if item then
      local take = r.GetActiveTake(item)
      if take and not r.TakeIsMIDI(take) then
        local src = r.GetMediaItemTake_Source(take)
        local src_path = r.GetMediaSourceFileName(src, "")
        if src_path and src_path ~= "" then
          local e = {
            item = item,
            take = take,
            path = src_path,
            pos = r.GetMediaItemInfo_Value(item, "D_POSITION") or 0,
            len = r.GetMediaItemInfo_Value(item, "D_LENGTH") or 0,
            start_off = r.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS") or 0,
            playrate = r.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE") or 1.0,
          }
          entries[#entries + 1] = e
          if not by_path[src_path] then
            by_path[src_path] = {}
            unique_paths[#unique_paths + 1] = src_path
          end
          by_path[src_path][#by_path[src_path] + 1] = e
        end
      end
    end
  end

  return entries, by_path, unique_paths
end

local function list_mode_string()
  if state.list_mode_idx == 5 then return "all" end
  if state.list_mode_idx == 4 then return "amphibia" end
  if state.list_mode_idx == 3 then return "insecta" end
  if state.list_mode_idx == 2 then return "mammalia" end
  if state.list_mode_idx == 1 then return "aves" end
  return "local"
end

local function engine_string()
  if state.engine_idx == 1 then return "birdnet" end
  return "perch_v2"
end

local function engine_display_name()
  if state.engine_idx == 1 then return "BirdNET" end
  return "Perch v2"
end

local function active_python_path()
  resolve_runtime_paths()
  if engine_string() == "birdnet" then
    return first_existing(birdnet_python_candidates())
  end
  return first_existing(perch_python_candidates())
end

local function build_request_json(unique_paths)
  local req = {}
  req[#req + 1] = '{"paths":['
  for i, p in ipairs(unique_paths) do
    req[#req + 1] = '"' .. json_escape(p) .. '"'
    if i < #unique_paths then req[#req + 1] = "," end
  end
  req[#req + 1] = string.format(
    '],"engine":"%s","top_k":%d,"name_threshold_rel":%.3f,"marker_threshold_rel":%.3f,"detections_per_segment":%d,"name_top_n":%d,"name_as_list":%s,"list_mode":"%s","strict_local":%s,"latitude":"%s","longitude":"%s","perch_model":"%s","perch_labels":"%s","local_birds_file":"%s","marker_gap_s":%.3f,"onset_snap":%s,"onset_lookback_s":%.3f,"onset_lookahead_s":%.3f,"birdnet_max_audio_min":%.3f}',
    json_escape(engine_string()),
    state.top_k,
    state.name_threshold_pct,
    state.marker_threshold_pct,
    state.detections_per_segment,
    state.name_top_n,
    state.name_as_list and "true" or "false",
    json_escape(list_mode_string()),
    state.strict_local and "true" or "false",
    json_escape(state.latitude),
    json_escape(state.longitude),
    json_escape(PERCH_MODEL_PATH),
    json_escape(PERCH_LABELS_PATH),
    json_escape(LOCAL_BIRDS_FILE),
    state.marker_gap_s,
    state.onset_snap and "true" or "false",
    state.onset_lookback_s,
    state.onset_lookahead_s,
    state.birdnet_max_audio_min
  )
  return table.concat(req)
end

local function get_last_log_tail(max_lines)
  max_lines = max_lines or 10
  if not file_exists(LOG_FILE) then return "" end
  local f = io.open(LOG_FILE, "r")
  if not f then return "" end
  local lines = {}
  for line in f:lines() do
    lines[#lines + 1] = line
    if #lines > max_lines then table.remove(lines, 1) end
  end
  f:close()
  return table.concat(lines, "\n")
end

local function count_species_lines(path)
  local f = io.open(path, "r")
  if not f then return 0 end
  local n = 0
  for line in f:lines() do
    local t = trim(line)
    if t ~= "" and t:sub(1, 1) ~= "#" then
      n = n + 1
    end
  end
  f:close()
  return n
end

local function launch_local_list_update()
  resolve_runtime_paths()
  local updater_path = active_ebird_updater_path()
  local python_path = first_existing(updater_python_candidates())
  local api_key = trim(state.ebird_api_key)
  local lat = tonumber(trim(state.latitude))
  local lon = tonumber(trim(state.longitude))
  local radius_km = math.max(1, math.min(500, math.floor(tonumber(state.ebird_radius_km) or 50)))
  local back_days = math.max(1, math.min(365, math.floor(tonumber(state.ebird_back_days) or 30)))
  local max_results = math.max(100, math.min(20000, math.floor(tonumber(state.ebird_max_results) or 10000)))

  if not lat or not lon then
    state.status_line = "Provide valid latitude and longitude first."
    state.last_summary = "Set Latitude/Longitude in Expert tab before updating local birds list."
    return
  end
  if api_key == "" then
    state.status_line = "eBird API key missing."
    state.last_summary = "Set eBird API key in Expert tab."
    return
  end
  if not file_exists(updater_path) then
    state.status_line = "eBird updater script not found."
    state.last_summary = "Expected updater:\n" .. updater_path
    return
  end
  if not file_exists(python_path) then
    state.status_line = "Python not found for local list updater."
    state.last_summary = "Set Perch/BirdNET python override or install Python."
    return
  end

  local out_file = trim(state.local_birds_override)
  if out_file == "" then
    out_file = active_bundle_dir() .. SEP .. "species_list.txt"
    state.local_birds_override = out_file
  end

  local cmd = shell_quote(python_path) .. " " .. shell_quote(updater_path)
    .. " --lat " .. tostring(lat)
    .. " --lon " .. tostring(lon)
    .. " --radius-km " .. tostring(radius_km)
    .. " --back-days " .. tostring(back_days)
    .. " --max-results " .. tostring(max_results)
    .. " --api-key " .. shell_quote(api_key)
    .. " --out-file " .. shell_quote(out_file)
    .. " >" .. shell_quote(LOG_FILE) .. " 2>&1"

  state.status_line = "Updating local list from eBird..."
  local ok = os.execute(cmd)
  if ok == false or not file_exists(out_file) then
    state.status_line = "Local list update failed."
    local tail = get_last_log_tail(16)
    state.last_summary = "Updater failed.\n\nLog tail:\n" .. (tail ~= "" and tail or "(empty)")
    return
  end

  local species_count = count_species_lines(out_file)
  save_path_overrides()
  resolve_runtime_paths()
  state.status_line = "Local list updated from eBird."
  state.last_summary = table.concat({
    "Updated local birds list:",
    out_file,
    "",
    "Latitude: " .. tostring(lat),
    "Longitude: " .. tostring(lon),
    "Radius (km): " .. tostring(radius_km),
    "Back days: " .. tostring(back_days),
    "Max observations: " .. tostring(max_results),
    "Species lines: " .. tostring(species_count),
    "",
    "Log: " .. LOG_FILE,
  }, "\n")
end

local function run_analysis()
  resolve_runtime_paths()
  local python_path = active_python_path()
  local helper_path = active_helper_path()

  if not file_exists(python_path) then
    if engine_string() == "birdnet" then
      state.status_line = "BirdNET Python env not found."
    else
      state.status_line = "Perch Python venv not found."
    end
    state.last_summary = "Run setup from Terminal:\n" .. setup_command_hint()
    return
  end
  if not file_exists(helper_path) then
    state.status_line = "Detection helper not found."
    state.last_summary = "Expected helper:\n" .. helper_path
    return
  end
  if engine_string() == "perch_v2" then
    if not file_exists(PERCH_MODEL_PATH) then
      state.status_line = "Perch model not found."
      state.last_summary = "Expected model:\n" .. PERCH_MODEL_PATH
      return
    end
    if not file_exists(PERCH_LABELS_PATH) then
      state.status_line = "Perch labels not found."
      state.last_summary = "Expected labels:\n" .. PERCH_LABELS_PATH
      return
    end
  end

  local entries, by_path, unique_paths = collect_selected_entries()
  if #entries == 0 then
    state.status_line = "Select one or more audio items first."
    return
  end

  local req_json = build_request_json(unique_paths)
  local rf = io.open(REQUEST_FILE, "w")
  if not rf then
    state.status_line = "Failed to write request file."
    return
  end
  rf:write(req_json)
  rf:close()

  os.remove(RESPONSE_FILE)

  local cmd = shell_quote(python_path) .. " " .. shell_quote(helper_path)
    .. " --request " .. shell_quote(REQUEST_FILE)
    .. " --response " .. shell_quote(RESPONSE_FILE)
    .. " --top-k " .. tostring(state.top_k)
    .. " >" .. shell_quote(LOG_FILE) .. " 2>&1"

  state.status_line = "Running " .. engine_display_name() .. "..."
  local ok = os.execute(cmd)

  if not file_exists(RESPONSE_FILE) then
    state.status_line = "No response from helper. Check log."
    local tail = get_last_log_tail(12)
    local helper_err = tail:match("ERROR:%s*([^\n]+)")
    if helper_err and helper_err ~= "" then
      state.status_line = helper_err
    end
    state.last_summary = "No response TSV.\n\nLog tail:\n" .. (tail ~= "" and tail or "(empty)")
    if ok == false then
      state.last_summary = state.last_summary .. "\n\nCommand failed."
    end
    return
  end

  local results = read_tsv_results(RESPONSE_FILE)
  if not results then
    state.status_line = "Failed reading response TSV."
    return
  end

  local renamed = 0
  local notes_written = 0
  local markers_added = 0
  local matched = 0
  local missing = 0
  local marker_offset_s = (state.marker_offset_ms or 0) / 1000.0

  r.Undo_BeginBlock()

  for path, items in pairs(by_path) do
    local rs = results[path]
    if not rs then
      local b = basename(path)
      for k, v in pairs(results) do
        if basename(k) == b then
          rs = v
          break
        end
      end
    end

    if not rs then
      missing = missing + #items
    else
      matched = matched + #items
      for _, e in ipairs(items) do
        if state.rename_take and e.take and rs.name ~= "" then
          r.GetSetMediaItemTakeInfo_String(e.take, "P_NAME", clamp_name(rs.name, state.max_take_name), true)
          renamed = renamed + 1
        end

        if state.write_notes and e.item then
          local note = {}
          note[#note + 1] = engine_display_name() .. " Detect (IMGUI)"
          note[#note + 1] = "Name: " .. (rs.name or "")
          note[#note + 1] = "SubCategory: " .. (rs.subcat or "")
          note[#note + 1] = "Action: " .. (rs.action or "")
          note[#note + 1] = "ContentType: " .. (rs.content_type or "")
          note[#note + 1] = string.format("Confidence: %.3f", rs.confidence or 0)
          note[#note + 1] = "ListMode: " .. (rs.list_mode_used ~= "" and rs.list_mode_used or list_mode_string())
          note[#note + 1] = "StrictLocal: " .. tostring(state.strict_local)
          note[#note + 1] = string.format("NameThreshold(Rel): %.2f%%", state.name_threshold_pct)
          note[#note + 1] = string.format("MarkerThreshold(Rel): %.2f%%", state.marker_threshold_pct)
          note[#note + 1] = "NameThresholdMode: " .. (rs.threshold_mode ~= "" and rs.threshold_mode or "relative")
          note[#note + 1] = "MarkerThresholdMode: " .. (rs.marker_threshold_mode ~= "" and rs.marker_threshold_mode or "relative")
          note[#note + 1] = string.format("RawTop1: %.6f", rs.raw_top1 or 0)
          note[#note + 1] = string.format("RawNameCutoff: %.6f", rs.name_raw_cutoff or 0)
          note[#note + 1] = string.format("RawMarkerCutoff: %.6f", rs.marker_raw_cutoff or 0)
          note[#note + 1] = "DetectionsPerSegment: " .. tostring(state.detections_per_segment)
          note[#note + 1] = "NameTopN: " .. tostring(state.name_top_n)
          note[#note + 1] = string.format("MarkerGap: %.2fs", state.marker_gap_s)
          note[#note + 1] = "OnsetSnap: " .. tostring(state.onset_snap)
          note[#note + 1] = string.format("OnsetWindow: -%.2fs/+%.2fs", state.onset_lookback_s, state.onset_lookahead_s)
          note[#note + 1] = "Top Matches:"
          note[#note + 1] = "1) " .. (rs.top1 ~= "" and rs.top1 or "-")
          if rs.top1_desc and rs.top1_desc ~= "" then note[#note + 1] = "   Desc: " .. rs.top1_desc end
          note[#note + 1] = "2) " .. (rs.top2 ~= "" and rs.top2 or "-")
          if rs.top2_desc and rs.top2_desc ~= "" then note[#note + 1] = "   Desc: " .. rs.top2_desc end
          note[#note + 1] = "3) " .. (rs.top3 ~= "" and rs.top3 or "-")
          if rs.top3_desc and rs.top3_desc ~= "" then note[#note + 1] = "   Desc: " .. rs.top3_desc end
          r.GetSetMediaItemInfo_String(e.item, "P_NOTES", table.concat(note, "\n"), true)
          notes_written = notes_written + 1
        end

        if state.add_markers and e.item then
          local added_for_item = 0
          local scenes = parse_scene_markers(rs.scene_markers)
          local rate = tonumber(e.playrate) or 1.0
          if rate <= 0 then rate = 1.0 end

          for _, sc in ipairs(scenes) do
            local rel = ((sc.t or 0) - (e.start_off or 0)) / rate
            if rel >= -0.001 and rel <= (e.len or 0) + 0.001 then
              local marker_pos = (e.pos or 0) + rel + marker_offset_s
              local marker_name = sanitize_marker_text(sc.label)
              if marker_name ~= "" then
                r.AddProjectMarker2(0, false, marker_pos, 0, marker_name, -1, 0)
                markers_added = markers_added + 1
                added_for_item = added_for_item + 1
              end
            end
          end

          if added_for_item == 0 and state.add_fallback_marker then
            local fallback_name = sanitize_marker_text(rs.name)
            if fallback_name ~= "" then
              r.AddProjectMarker2(0, false, (e.pos or 0) + marker_offset_s, 0, fallback_name, -1, 0)
              markers_added = markers_added + 1
            end
          end
        end
      end
    end
  end

  r.UpdateArrange()
  r.Undo_EndBlock("SGM Perch and Birdnet Detect", -1)

  local summary = {}
  summary[#summary + 1] = "Processed items: " .. tostring(#entries)
  summary[#summary + 1] = "Engine: " .. engine_string()
  summary[#summary + 1] = "List mode: " .. list_mode_string()
  summary[#summary + 1] = "Strict local: " .. tostring(state.strict_local)
  summary[#summary + 1] = string.format("Name threshold (relative): %.2f%%", state.name_threshold_pct)
  summary[#summary + 1] = string.format("Marker threshold (relative): %.2f%%", state.marker_threshold_pct)
  summary[#summary + 1] = "Detections/segment: " .. tostring(state.detections_per_segment)
  summary[#summary + 1] = "Name top-N: " .. tostring(state.name_top_n)
  summary[#summary + 1] = string.format("Marker gap: %.2fs", state.marker_gap_s)
  summary[#summary + 1] = "Onset snap: " .. tostring(state.onset_snap)
  summary[#summary + 1] = string.format("Onset window: -%.2fs/+%.2fs", state.onset_lookback_s, state.onset_lookahead_s)
  summary[#summary + 1] = string.format("Marker offset: %d ms", state.marker_offset_ms)
  summary[#summary + 1] = "Python: " .. python_path
  summary[#summary + 1] = "Helper: " .. helper_path
  summary[#summary + 1] = "Local list: " .. LOCAL_BIRDS_FILE
  if engine_string() == "perch_v2" then
    summary[#summary + 1] = "Perch model: " .. PERCH_MODEL_PATH
    summary[#summary + 1] = "Perch labels: " .. PERCH_LABELS_PATH
  end
  summary[#summary + 1] = "Renamed takes: " .. tostring(renamed)
  summary[#summary + 1] = "Item notes written: " .. tostring(notes_written)
  summary[#summary + 1] = "Action markers added: " .. tostring(markers_added)
  summary[#summary + 1] = "Matched: " .. tostring(matched)
  summary[#summary + 1] = "No suggestion: " .. tostring(missing)
  summary[#summary + 1] = ""
  summary[#summary + 1] = "Log: " .. LOG_FILE

  state.last_summary = table.concat(summary, "\n")
  state.status_line = "Done."
end

local function draw_standard_tab(ctx)
  local changed
  resolve_runtime_paths()

  changed, state.engine_idx = r.ImGui_Combo(ctx, "Engine", state.engine_idx, ENGINE_ITEMS)
  changed, state.name_threshold_pct = r.ImGui_SliderDouble(ctx, "Name threshold (%)", state.name_threshold_pct, 0.0, 100.0, "%.2f%%")
  changed, state.marker_threshold_pct = r.ImGui_SliderDouble(ctx, "Marker threshold (%)", state.marker_threshold_pct, 0.0, 100.0, "%.2f%%")
  changed, state.detections_per_segment = r.ImGui_SliderInt(ctx, "Detections per segment", state.detections_per_segment, 1, 10)
  changed, state.name_top_n = r.ImGui_SliderInt(ctx, "Name top-N labels", state.name_top_n, 1, 10)
  changed, state.list_mode_idx = r.ImGui_Combo(ctx, "List", state.list_mode_idx, LIST_MODE_ITEMS)
  changed, state.name_as_list = r.ImGui_Checkbox(ctx, "Use top-N list in take name", state.name_as_list)

  changed, state.rename_take = r.ImGui_Checkbox(ctx, "Rename take names", state.rename_take)
  changed, state.write_notes = r.ImGui_Checkbox(ctx, "Write item notes", state.write_notes)
  changed, state.add_markers = r.ImGui_Checkbox(ctx, "Add timeline markers", state.add_markers)
  changed, state.add_fallback_marker = r.ImGui_Checkbox(ctx, "Fallback marker when none detected", state.add_fallback_marker)
  if state.add_markers then
    changed, state.onset_snap = r.ImGui_Checkbox(ctx, "Snap marker to call onset", state.onset_snap)
  end
  if state.list_mode_idx == 0 then
    changed, state.strict_local = r.ImGui_Checkbox(ctx, "Strict local list (no aves fallback)", state.strict_local)
  end

  local lm = list_mode_string()
  if state.engine_idx == 1 then
    if lm == "local" then
      text_colored(ctx, UI_COLORS.accent_sun, "Local list mode uses: " .. LOCAL_BIRDS_FILE)
      text_colored(ctx, UI_COLORS.accent_sky, "BirdNET local mode filters bird species by your local list.")
    elseif lm == "mammalia" or lm == "insecta" or lm == "amphibia" then
      text_colored(ctx, UI_COLORS.accent_sky, "BirdNET is bird-only. This mode falls back to Aves.")
    else
      text_colored(ctx, UI_COLORS.accent_sky, "BirdNET is bird-only. In BirdNET, 'Aves' and 'All Labels' behave the same.")
    end
  elseif lm == "all" then
    text_colored(ctx, UI_COLORS.accent_leaf, "All Labels mode can return birds and non-bird animals if the model has those labels.")
  elseif lm == "local" then
    text_colored(ctx, UI_COLORS.accent_sun, "Local list mode uses: " .. LOCAL_BIRDS_FILE)
  else
    text_colored(ctx, UI_COLORS.accent_leaf, "Perch group mode filters labels by taxonomy group.")
  end
  text_colored(ctx, UI_COLORS.text_dim, "Bundle folder: " .. active_bundle_dir())
  text_colored(ctx, UI_COLORS.text_dim, "Helper path: " .. active_helper_path())
  r.ImGui_TextWrapped(ctx, "Thresholds are RELATIVE (0-100) against the file's top score from the selected engine. Raw score values are written into item notes.")
end

local function prompt_text_path(title, label, current)
  local ok, ret = r.GetUserInputs(title, 1, label .. ",extrawidth=280", current or "")
  if ok then return trim(ret) end
  return nil
end

local function parent_dir(path)
  path = trim(path)
  if path == "" then return "" end
  return (path:match("^(.*)[/\\][^/\\]+$")) or ""
end

local function pick_file_path(title, current, ext)
  local seed = trim(current)
  if seed == "" then seed = active_bundle_dir() end
  local ok, picked = r.GetUserFileNameForRead(seed, title, ext or "")
  if ok then return trim(picked) end
  return nil
end

local function pick_folder_path(title, current)
  local seed = trim(current)
  if seed == "" then seed = active_bundle_dir() end
  local ok, picked = r.GetUserFileNameForRead(seed, title .. " (pick any file inside folder)", "")
  if not ok then return nil end
  local dir = parent_dir(picked)
  if dir ~= "" then return dir end
  return nil
end

local function trim_trailing_sep(path)
  path = trim(path)
  while #path > 1 do
    local last = path:sub(-1)
    if last ~= "/" and last ~= "\\" then break end
    path = path:sub(1, -2)
  end
  return path
end

local function path_under_dir(path, dir)
  path = trim_trailing_sep(path)
  dir = trim_trailing_sep(dir)
  if path == "" or dir == "" then return false end
  if path == dir then return true end
  if path:sub(1, #dir) ~= dir then return false end
  local next_char = path:sub(#dir + 1, #dir + 1)
  return next_char == "/" or next_char == "\\"
end

local function auto_fill_overrides_from_bundle(old_bundle_dir, new_bundle_dir)
  old_bundle_dir = trim_trailing_sep(old_bundle_dir)
  new_bundle_dir = trim_trailing_sep(new_bundle_dir)
  if new_bundle_dir == "" then return false end

  local changed = false
  local setup_path = first_existing({
    new_bundle_dir .. SEP .. "setup_perch_birdnet_macos.sh",
    new_bundle_dir .. SEP .. "setup_macos.sh",
  })
  local perch_model = new_bundle_dir .. SEP .. "models" .. SEP .. "perch_v2" .. SEP .. "perch_v2.onnx"
  local perch_labels = new_bundle_dir .. SEP .. "models" .. SEP .. "perch_v2" .. SEP .. "labels.txt"
  local local_birds = new_bundle_dir .. SEP .. "species_list.txt"
  local perch_python = first_existing({
    new_bundle_dir .. SEP .. "venv" .. SEP .. "perch_py312" .. SEP .. "bin" .. SEP .. "python3.12",
    new_bundle_dir .. SEP .. "venv" .. SEP .. "perch_py312" .. SEP .. "bin" .. SEP .. "python3",
  })
  local birdnet_python = first_existing({
    new_bundle_dir .. SEP .. "venv" .. SEP .. "birdnet_arm64" .. SEP .. "bin" .. SEP .. "python3",
  })

  local function maybe_update(field, new_value)
    local cur = trim(state[field] or "")
    local can_replace = (cur == "")
      or (old_bundle_dir ~= "" and path_under_dir(cur, old_bundle_dir))
    if can_replace and cur ~= new_value then
      state[field] = new_value
      changed = true
    end
  end

  maybe_update("setup_script_override", setup_path)
  maybe_update("perch_model_override", perch_model)
  maybe_update("perch_labels_override", perch_labels)
  maybe_update("local_birds_override", local_birds)
  maybe_update("perch_python_override", perch_python)
  maybe_update("birdnet_python_override", birdnet_python)

  return changed
end

local function draw_expert_tab(ctx)
  local changed
  local path_changed = false
  local bundle_changed = false
  local old_bundle_dir = active_bundle_dir()
  changed, state.marker_gap_s = r.ImGui_SliderDouble(ctx, "Marker gap (sec)", state.marker_gap_s, 0.0, 20.0, "%.2f")
  changed, state.onset_lookback_s = r.ImGui_SliderDouble(ctx, "Onset lookback (sec)", state.onset_lookback_s, 0.00, 2.00, "%.2f")
  changed, state.onset_lookahead_s = r.ImGui_SliderDouble(ctx, "Onset lookahead (sec)", state.onset_lookahead_s, 0.10, 3.00, "%.2f")
  changed, state.marker_offset_ms = r.ImGui_SliderInt(ctx, "Marker offset (ms)", state.marker_offset_ms, -3000, 3000)
  changed, state.top_k = r.ImGui_SliderInt(ctx, "Top matches in note", state.top_k, 3, 10)
  changed, state.max_take_name = r.ImGui_SliderInt(ctx, "Max take-name length", state.max_take_name, 40, 240)

  changed, state.latitude = r.ImGui_InputText(ctx, "Latitude (optional)", state.latitude)
  changed, state.longitude = r.ImGui_InputText(ctx, "Longitude (optional)", state.longitude)
  if state.engine_idx == 1 then
    changed, state.birdnet_max_audio_min = r.ImGui_SliderDouble(ctx, "BirdNET max audio (min)", state.birdnet_max_audio_min, 0.50, 20.00, "%.2f")
  end

  r.ImGui_Separator(ctx)
  text_colored(ctx, UI_COLORS.accent_leaf, "eBird local-list updater")
  changed, state.ebird_api_key = r.ImGui_InputText(ctx, "eBird API key", state.ebird_api_key)
  if changed then path_changed = true end
  changed, state.ebird_radius_km = r.ImGui_SliderInt(ctx, "eBird radius (km)", math.floor(tonumber(state.ebird_radius_km) or 50), 1, 500)
  if changed then path_changed = true end
  changed, state.ebird_back_days = r.ImGui_SliderInt(ctx, "eBird lookback days", math.floor(tonumber(state.ebird_back_days) or 30), 1, 365)
  if changed then path_changed = true end
  changed, state.ebird_max_results = r.ImGui_SliderInt(ctx, "eBird max observations", math.floor(tonumber(state.ebird_max_results) or 10000), 100, 20000)
  if changed then path_changed = true end
  if r.ImGui_Button(ctx, "Update local list now (eBird)") then
    launch_local_list_update()
  end

  r.ImGui_Separator(ctx)
  text_colored(ctx, UI_COLORS.accent_sky, "Runtime paths (optional overrides)")
  r.ImGui_TextWrapped(ctx, "Set these when your setup/model/species files are in custom folders, or when BirdNET is already installed in another Python environment.")

  changed, state.bundle_dir_override = r.ImGui_InputText(ctx, "Bundle folder override", state.bundle_dir_override)
  if changed then
    path_changed = true
    bundle_changed = true
  end
  r.ImGui_SameLine(ctx)
  if r.ImGui_Button(ctx, "Browse##bundle") then
    local picked = pick_folder_path("Choose bundle folder", state.bundle_dir_override)
    if picked ~= nil then
      state.bundle_dir_override = picked
      path_changed = true
      bundle_changed = true
    end
  end
  r.ImGui_SameLine(ctx)
  if r.ImGui_Button(ctx, "Type##bundle") then
    local picked = prompt_text_path("Bundle folder override", "Folder path", state.bundle_dir_override)
    if picked ~= nil then
      state.bundle_dir_override = picked
      path_changed = true
      bundle_changed = true
    end
  end

  if bundle_changed then
    local new_bundle_dir = active_bundle_dir()
    if auto_fill_overrides_from_bundle(old_bundle_dir, new_bundle_dir) then
      path_changed = true
    end
  end

  changed, state.setup_script_override = r.ImGui_InputText(ctx, "Setup script override", state.setup_script_override)
  if changed then path_changed = true end
  r.ImGui_SameLine(ctx)
  if r.ImGui_Button(ctx, "Pick##setup_script") then
    local picked = pick_file_path("Select setup shell script", state.setup_script_override, ".sh")
    if picked then
      state.setup_script_override = picked
      path_changed = true
    end
  end

  changed, state.helper_path_override = r.ImGui_InputText(ctx, "Helper script override", state.helper_path_override)
  if changed then path_changed = true end
  r.ImGui_SameLine(ctx)
  if r.ImGui_Button(ctx, "Pick##helper_script") then
    local picked = pick_file_path("Select sfx_perch_auto_name_helper.py", state.helper_path_override, ".py")
    if picked then
      state.helper_path_override = picked
      path_changed = true
    end
  end

  changed, state.ebird_updater_override = r.ImGui_InputText(ctx, "eBird updater override", state.ebird_updater_override)
  if changed then path_changed = true end
  r.ImGui_SameLine(ctx)
  if r.ImGui_Button(ctx, "Pick##ebird_updater") then
    local picked = pick_file_path("Select sfx_ebird_local_list_updater.py", state.ebird_updater_override, ".py")
    if picked then
      state.ebird_updater_override = picked
      path_changed = true
    end
  end

  changed, state.perch_model_override = r.ImGui_InputText(ctx, "Perch model override", state.perch_model_override)
  if changed then path_changed = true end
  r.ImGui_SameLine(ctx)
  if r.ImGui_Button(ctx, "Pick##perch_model") then
    local picked = pick_file_path("Select perch_v2.onnx", state.perch_model_override, ".onnx")
    if picked then
      state.perch_model_override = picked
      path_changed = true
    end
  end

  changed, state.perch_labels_override = r.ImGui_InputText(ctx, "Perch labels override", state.perch_labels_override)
  if changed then path_changed = true end
  r.ImGui_SameLine(ctx)
  if r.ImGui_Button(ctx, "Pick##perch_labels") then
    local picked = pick_file_path("Select labels.txt", state.perch_labels_override, ".txt")
    if picked then
      state.perch_labels_override = picked
      path_changed = true
    end
  end

  changed, state.local_birds_override = r.ImGui_InputText(ctx, "Local/species list override", state.local_birds_override)
  if changed then path_changed = true end
  r.ImGui_SameLine(ctx)
  if r.ImGui_Button(ctx, "Pick##local_birds") then
    local picked = pick_file_path("Select local/species list text file", state.local_birds_override, ".txt")
    if picked then
      state.local_birds_override = picked
      path_changed = true
    end
  end

  changed, state.perch_python_override = r.ImGui_InputText(ctx, "Perch python override", state.perch_python_override)
  if changed then path_changed = true end
  r.ImGui_SameLine(ctx)
  if r.ImGui_Button(ctx, "Pick##perch_python") then
    local picked = pick_file_path("Select Perch python executable", state.perch_python_override, "")
    if picked then
      state.perch_python_override = picked
      path_changed = true
    end
  end

  changed, state.birdnet_python_override = r.ImGui_InputText(ctx, "BirdNET python override", state.birdnet_python_override)
  if changed then path_changed = true end
  r.ImGui_SameLine(ctx)
  if r.ImGui_Button(ctx, "Pick##birdnet_python") then
    local picked = pick_file_path("Select BirdNET python executable", state.birdnet_python_override, "")
    if picked then
      state.birdnet_python_override = picked
      path_changed = true
    end
  end

  if r.ImGui_Button(ctx, "Clear path overrides") then
    clear_path_overrides()
    path_changed = true
  end

  if path_changed then
    save_path_overrides()
    resolve_runtime_paths()
  end

  resolve_runtime_paths()
  text_colored(ctx, UI_COLORS.text_dim, "Resolved setup: " .. active_setup_script())
  text_colored(ctx, UI_COLORS.text_dim, "Resolved eBird updater: " .. active_ebird_updater_path())
  text_colored(ctx, UI_COLORS.text_dim, "Resolved local/species list: " .. LOCAL_BIRDS_FILE)
  text_colored(ctx, UI_COLORS.text_dim, "Resolved Perch model: " .. PERCH_MODEL_PATH)
  text_colored(ctx, UI_COLORS.text_dim, "Resolved Perch labels: " .. PERCH_LABELS_PATH)
  text_colored(ctx, UI_COLORS.text_dim, "Resolved Perch python: " .. first_existing(perch_python_candidates()))
  text_colored(ctx, UI_COLORS.text_dim, "Resolved BirdNET python: " .. first_existing(birdnet_python_candidates()))
end

local ctx = r.ImGui_CreateContext(SCRIPT_TITLE)
local FLT_MIN = r.ImGui_NumericLimits_Float and r.ImGui_NumericLimits_Float() or 0.0

local function loop()
  r.ImGui_SetNextWindowSize(ctx, 640, 620, r.ImGui_Cond_FirstUseEver())
  local theme_applied = pcall(push_vibrant_theme, ctx)
  local visible, open = r.ImGui_Begin(ctx, SCRIPT_TITLE, true)
  if visible then
    local selected_count = r.CountSelectedMediaItems(0)
    draw_header(ctx, selected_count)
    r.ImGui_Separator(ctx)

    if r.ImGui_BeginTabBar and r.ImGui_BeginTabBar(ctx, "PerchTabs") then
      if r.ImGui_BeginTabItem(ctx, "Standard") then
        draw_standard_tab(ctx)
        r.ImGui_EndTabItem(ctx)
      end
      if r.ImGui_BeginTabItem(ctx, "Expert") then
        draw_expert_tab(ctx)
        r.ImGui_EndTabItem(ctx)
      end
      r.ImGui_EndTabBar(ctx)
    else
      draw_standard_tab(ctx)
      r.ImGui_Separator(ctx)
      draw_expert_tab(ctx)
    end

    r.ImGui_Separator(ctx)
    if r.ImGui_Button(ctx, "Run Setup (macOS)", -FLT_MIN, 0.0) then
      launch_setup_script()
    end
    r.ImGui_Separator(ctx)
    if r.ImGui_Button(ctx, "Analyze Selected", -FLT_MIN, 0.0) then
      run_analysis()
    end
    text_colored(ctx, status_color(state.status_line), "Status: " .. (state.status_line or ""))
    r.ImGui_Separator(ctx)
    if state.last_summary ~= "" then
      r.ImGui_TextWrapped(ctx, state.last_summary)
    else
      r.ImGui_TextWrapped(ctx, "Run analysis to see summary here.")
    end
    r.ImGui_End(ctx)
  end
  if theme_applied then
    pop_vibrant_theme(ctx)
  end

  if open then
    r.defer(loop)
  else
    if r.ImGui_DestroyContext then
      pcall(r.ImGui_DestroyContext, ctx)
    end
  end
end

r.defer(loop)
