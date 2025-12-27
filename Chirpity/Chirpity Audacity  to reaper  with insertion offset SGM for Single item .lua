-- @title        Import Annotation TXT as Regions (Cursor Offset)
-- @author       Sruthin
-- @version      1.0.0
-- @about
--   Imports annotation TXT files and creates REAPER regions
--   starting at the current edit cursor position.
--   Commonly used for Chirpity, Audacity, BirdNET, or similar
--   annotation exports.
--
-- @changelog
--   v1.0.0
--   - Initial release
--   - Cursor-based region offset
--   - Automatic label cleanup
------------------------------------------------------------


-- Converts time in seconds to timecode (HH:MM:SS:FF), 25 FPS
local function secondsToTimecode(seconds)
  local fps = 25
  local h = math.floor(seconds / 3600)
  local m = math.floor((seconds % 3600) / 60)
  local s = math.floor(seconds % 60)
  local f = math.floor((seconds - math.floor(seconds)) * fps)
  return string.format("%02d:%02d:%02d:%02d", h, m, s, f)
end

-- Cleans label by removing trailing confidence percentage
-- Example: "Species 84%" → "Species"
local function cleanLabel(label)
  return label:gsub("%s*%d+%.?%d*%%?$", ""):gsub("%s+$", "")
end

-- Process input file and add regions to REAPER
local function processFile(filePath)
  local file = io.open(filePath, "r")
  if not file then
    reaper.ShowConsoleMsg("❌ Cannot open file: " .. filePath .. "\n")
    return
  end

  local cursor_pos = reaper.GetCursorPosition()
  local color = tonumber("0xFBFF4B") -- Yellow
  local count = 0

  for line in file:lines() do
    local start_s, end_s, raw_label =
      line:match("([%d%.]+)%s+([%d%.]+)%s+(.+)")
    if start_s and end_s and raw_label then
      local name = cleanLabel(raw_label)
      local start_time = tonumber(start_s) + cursor_pos
      local end_time   = tonumber(end_s) + cursor_pos

      if start_time and end_time then
        reaper.AddProjectMarker2(
          0, true,
          start_time, end_time,
          name, -1, color
        )
        count = count + 1
      end
    end
  end

  file:close()
  reaper.ShowConsoleMsg(
    "✅ Imported " .. count .. " regions starting at cursor position.\n"
  )
end

-- Main
local function main()
  local retval, filePath =
    reaper.GetUserFileNameForRead(
      "",
      "Select region definition file",
      "*.txt"
    )

  if retval then
    reaper.Undo_BeginBlock()
    processFile(filePath)
    reaper.Undo_EndBlock(
      "Import annotation regions (offset from cursor)",
      -1
    )
  else
    reaper.ShowConsoleMsg("⚠️ No file selected.\n")
  end
end

main()
