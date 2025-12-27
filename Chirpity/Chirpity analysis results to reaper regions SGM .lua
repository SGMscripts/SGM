-- @title        Import Per-Item Annotation TXT as Regions (Item Offset)
-- @author       Sruthin
-- @version      1.0.0
-- @about
--   Imports annotation TXT files as regions for each selected item.
--   Each selected item looks for a matching TXT file (by item name)
--   in the same directory and inserts regions offset to the item start.
--
-- @changelog
--   v1.0.0
--   - Initial release
--   - Per-item annotation import
--   - Automatic label cleanup
------------------------------------------------------------


-- Clean label by removing trailing confidence percentage
-- Example: "Species 84%" → "Species"
local function cleanLabel(label)
  return label:gsub("%s*%d+%.?%d*%%?$", ""):gsub("%s+$", "")
end

-- Remove extension from a string
-- Example: "Bird.wav" → "Bird"
local function removeExtension(filename)
  return filename:match("^(.*)%.") or filename
end

-- Process a single TXT file for one item
local function processFileForItem(filePath, item)
  local file = io.open(filePath, "r")
  if not file then
    reaper.ShowConsoleMsg("❌ Cannot open file: " .. filePath .. "\n")
    return 0
  end

  local itemStart = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
  local color = tonumber("0xFBFF4B") -- Yellow
  local count = 0

  for line in file:lines() do
    local start_s, end_s, raw_label =
      line:match("([%d%.]+)%s+([%d%.]+)%s+(.+)")
    if start_s and end_s and raw_label then
      local name = cleanLabel(raw_label)
      local start_time = tonumber(start_s) + itemStart
      local end_time   = tonumber(end_s) + itemStart

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
  return count
end

-- Main
local function main()
  local itemCount = reaper.CountSelectedMediaItems(0)
  if itemCount == 0 then
    reaper.ShowConsoleMsg("⚠️ No items selected.\n")
    return
  end

  -- Pick any TXT file to detect directory
  local retval, pickedFile =
    reaper.GetUserFileNameForRead(
      "", "Select ANY annotation txt file (for directory)", "*.txt"
    )
  if not retval then
    reaper.ShowConsoleMsg("⚠️ No file selected.\n")
    return
  end

  local dir = pickedFile:match("^(.*[\\/])")
  if not dir then
    reaper.ShowConsoleMsg(
      "❌ Could not determine directory from: " .. pickedFile .. "\n"
    )
    return
  end

  reaper.Undo_BeginBlock()

  local totalCount = 0
  for i = 0, itemCount - 1 do
    local item = reaper.GetSelectedMediaIte
