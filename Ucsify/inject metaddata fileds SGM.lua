-- @description UCS Inject Render Metadata (Standalone)
-- @author You
-- @version 1.1.0
-- @about
--   Pre-populates Project Render Metadata with all UCS / Extended / ASWG keys
--   (same mapping as acendan_UCS Renaming Tool) without renaming anything.
--   Run before opening the Render dialog so fields aren’t empty.
--
--   By default writes $marker(Field)[;] substitutions (dynamic).
--   Optional: set direct_values = true to write current extstate literal values instead.
--
-- Requirements: REAPER 6.33+ for [;] syntax (falls back automatically).

--------------------------------------------------------
-- USER OPTIONS
--------------------------------------------------------
local direct_values = false      -- false = $marker() dynamic; true = write literal values (if present)
local always_overwrite = false   -- true = overwrite existing entries, false = only add missing keys
local embedder_name = "REAPER UCS Renaming Tool"

--------------------------------------------------------
-- Helpers
--------------------------------------------------------
local function getExt(name)
  local ret, val = reaper.GetProjExtState(0,"UCS_WebInterface",name)
  if ret == 1 then return val else return "" end
end

local function reaperVersionOK()
  return (tonumber(reaper.GetAppVersion():match("%d+%.%d+")) or 0) >= 6.33
end

local have633 = reaperVersionOK()

local function setMeta(metaKey, fieldName, literal)
  -- Skip if not overwriting and already exists
  if not always_overwrite then
    local _, blob = reaper.GetSetProjectInfo_String(0,"RENDER_METADATA","",false)
    if blob:find("^"..metaKey.."|") or blob:find("\n"..metaKey.."|") then return end
  end

  -- Special literals
  if fieldName == "ReleaseDate" then
    reaper.GetSetProjectInfo_String(0,"RENDER_METADATA", metaKey .. "|$date", true)
    return
  elseif fieldName == "Embedder" then
    reaper.GetSetProjectInfo_String(0,"RENDER_METADATA", metaKey .. "|" .. embedder_name, true)
    return
  elseif fieldName == "ASWGsession" then
    reaper.GetSetProjectInfo_String(0,"RENDER_METADATA", metaKey .. "|$project", true)
    return
  end

  if direct_values then
    reaper.GetSetProjectInfo_String(0,"RENDER_METADATA", metaKey .. "|" .. (literal or ""), true)
  else
    local suffix = have633 and "[;]" or ""
    reaper.GetSetProjectInfo_String(
      0,
      "RENDER_METADATA",
      metaKey .. "|$marker(" .. fieldName .. ")" .. suffix,
      true
    )
  end
end

--------------------------------------------------------
-- Collect literal values (only used if direct_values = true)
--------------------------------------------------------
local vals = {}
local function grab(field, extKey) vals[field] = getExt(extKey) end

-- Core UCS / Extended
grab("CatID","CatID")
grab("Category","Category")
grab("SubCategory","Subcategory")
grab("FXName","Name")
grab("Notes","Data")
grab("Show","Show")
grab("UserCategory","UserCategory")
grab("VendorCategory","VendorCategory")
grab("TrackTitle","MetaTitle")
grab("Description","MetaDesc")
grab("Keywords","MetaKeys")
grab("Microphone","MetaMic")
grab("MicPerspective","MetaPersp")
grab("RecType","MetaConfig")
grab("RecMedium","MetaRecMed")
grab("Designer","MetaDsgnr")
grab("Location","MetaLoc")
grab("URL","MetaURL")
grab("Manufacturer","MetaMftr")
grab("Library","MetaLib")
grab("MetaNotes","Notes")

-- NEW: Layout fields
grab("ChannelLayout","ChannelLayout")
grab("IxmlTrackLayout","IxmlTrackLayout")

--------------------------------------------------------
-- Derived fields
--------------------------------------------------------
if vals.Category ~= "" and vals.SubCategory ~= "" then
  vals.CategoryFull = vals.Category .. "-" .. vals.SubCategory
end

-- ShortID from Designer (3 chars per word)
do
  local d = vals.Designer or ""
  local short = ""
  for w in d:gmatch("%S+") do short = short .. w:sub(1,3) end
  vals.ShortID = short
end

vals.LongID  = vals.CatID
vals.Source  = vals.Show
vals.Artist  = vals.Designer

--------------------------------------------------------
-- ASWG fields
--------------------------------------------------------
local aswgList = {
  "ASWGcontentType","ASWGproject","ASWGoriginatorStudio","ASWGnotes","ASWGstate",
  "ASWGeditor","ASWGmixer","ASWGfxChainName","ASWGchannelConfig","ASWGambisonicFormat",
  "ASWGambisonicChnOrder","ASWGambisonicNorm","ASWGisDesigned","ASWGrecEngineer",
  "ASWGrecStudio","ASWGimpulseLocation","ASWGtext","ASWGefforts","ASWGeffortType",
  "ASWGprojection","ASWGlanguage","ASWGtimingRestriction","ASWGcharacterName",
  "ASWGcharacterGender","ASWGcharacterAge","ASWGcharacterRole","ASWGactorName",
  "ASWGactorGender","ASWGdirection","ASWGdirector","ASWGfxUsed","ASWGusageRights",
  "ASWGisUnion","ASWGaccent","ASWGemotion","ASWGcomposer","ASWGartist","ASWGsongTitle",
  "ASWGgenre","ASWGsubGenre","ASWGproducer","ASWGmusicSup","ASWGinstrument",
  "ASWGmusicPublisher","ASWGrightsOwner","ASWGintensity","ASWGorderRef","ASWGisSource",
  "ASWGisLoop","ASWGisFinal","ASWGisOst","ASWGisCinematic","ASWGisLicensed",
  "ASWGisDiegetic","ASWGmusicVersion","ASWGisrcId","ASWGtempo","ASWGtimeSig",
  "ASWGinKey","ASWGbillingCode"
}
for _, f in ipairs(aswgList) do vals[f] = getExt(f) end

--------------------------------------------------------
-- Mapping (metaKey => fieldName)
--------------------------------------------------------
local map = {
  -- IXML USER
  ["IXML:USER:CatID"]="CatID",
  ["IXML:USER:Category"]="Category",
  ["IXML:USER:SubCategory"]="SubCategory",
  ["IXML:USER:CategoryFull"]="CategoryFull",
  ["IXML:USER:FXName"]="FXName",
  ["IXML:USER:Notes"]="Notes",
  ["IXML:USER:Show"]="Show",
  ["IXML:USER:UserCategory"]="UserCategory",
  ["IXML:USER:VendorCategory"]="VendorCategory",
  ["IXML:USER:TrackTitle"]="TrackTitle",
  ["IXML:USER:Description"]="Description",
  ["IXML:USER:Keywords"]="Keywords",
  ["IXML:USER:Microphone"]="Microphone",
  ["IXML:USER:MicPerspective"]="MicPerspective",
  ["IXML:USER:RecType"]="RecType",
  ["IXML:USER:RecMedium"]="RecMedium",
  ["IXML:USER:Designer"]="Designer",
  ["IXML:USER:ShortID"]="ShortID",
  ["IXML:USER:Location"]="Location",
  ["IXML:USER:URL"]="URL",
  ["IXML:USER:Manufacturer"]="Manufacturer",
  ["IXML:USER:Library"]="Library",
  ["IXML:USER:ReleaseDate"]="ReleaseDate",
  ["IXML:USER:Embedder"]="Embedder",
  ["IXML:USER:LongID"]="CatID",
  ["IXML:USER:Source"]="Show",
  ["IXML:USER:Artist"]="Designer",

  -- NEW: Layout metadata
  ["IXML:USER:ChannelLayout"]     = "ChannelLayout",
  ["IXML:USER:IxmlTrackLayout"]   = "IxmlTrackLayout",

  -- BWF / INFO / ID3 / XMP / VORBIS
  ["IXML:BEXT:BWF_DESCRIPTION"]="Description",
  ["BWF:Description"]="Description",
  ["BWF:Originator"]="Designer",
  ["BWF:OriginatorReference"]="URL",
  ["BWF:ChannelLayout"]="ChannelLayout",

  ["INFO:ICMT"]="Description",
  ["INFO:IART"]="Designer",
  ["INFO:IGNR"]="Category",
  ["INFO:INAM"]="TrackTitle",
  ["INFO:IPRD"]="Library",
  ["INFO:IARL"]="IxmlTrackLayout",

  ["ID3:TIT2"]="TrackTitle",
  ["ID3:COMM"]="Description",
  ["ID3:TPE1"]="Designer",
  ["ID3:TPE2"]="Show",
  ["ID3:TCON"]="Category",
  ["ID3:TALB"]="Library",

  ["XMP:dc/description"]="Description",
  ["XMP:dm/artist"]="Designer",
  ["XMP:dm/genre"]="Category",
  ["XMP:dc/title"]="TrackTitle",
  ["XMP:dm/album"]="Library",

  ["VORBIS:DESCRIPTION"]="Description",
  ["VORBIS:COMMENT"]="Description",
  ["VORBIS:GENRE"]="Category",
  ["VORBIS:TITLE"]="TrackTitle",
  ["VORBIS:ARTIST"]="Designer",
  ["VORBIS:ALBUM"]="Library",

  -- ASWG session
  ["ASWG:session"]="ASWGsession"
}

--------------------------------------------------------
-- Add ASWG dynamic keys
--------------------------------------------------------
local aswgMap = {
  contentType="ASWGcontentType", project="ASWGproject", originatorStudio="ASWGoriginatorStudio",
  notes="ASWGnotes", state="ASWGstate", editor="ASWGeditor", mixer="ASWGmixer",
  fxChainName="ASWGfxChainName", channelConfig="ASWGchannelConfig",
  ambisonicFormat="ASWGambisonicFormat", ambisonicChnOrder="ASWGambisonicChnOrder",
  ambisonicNorm="ASWGambisonicNorm", isDesigned="ASWGisDesigned",
  recEngineer="ASWGrecEngineer", recStudio="ASWGrecStudio",
  impulseLocation="ASWGimpulseLocation", text="ASWGtext", efforts="ASWGefforts",
  effortType="ASWGeffortType", projection="ASWGprojection", language="ASWGlanguage",
  timingRestriction="ASWGtimingRestriction", characterName="ASWGcharacterName",
  characterGender="ASWGcharacterGender", characterAge="ASWGcharacterAge",
  characterRole="ASWGcharacterRole", actorName="ASWGactorName",
  actorGender="ASWGactorGender", direction="ASWGdirection",
  director="ASWGdirector", fxUsed="ASWGfxUsed",
  usageRights="ASWGusageRights", isUnion="ASWGisUnion",
  accent="ASWGaccent", emotion="ASWGemotion",
  composer="ASWGcomposer", artist="ASWGartist",
  songTitle="ASWGsongTitle", genre="ASWGgenre", subGenre="ASWGsubGenre",
  producer="ASWGproducer", musicSup="ASWGmusicSup",
  instrument="ASWGinstrument", musicPublisher="ASWGmusicPublisher",
  rightsOwner="ASWGrightsOwner", intensity="ASWGintensity",
  orderRef="ASWGorderRef", isSource="ASWGisSource",
  isLoop="ASWGisLoop", isFinal="ASWGisFinal",
  isOst="ASWGisOst", isCinematic="ASWGisCinematic",
  isLicensed="ASWGisLicensed", isDiegetic="ASWGisDiegetic",
  musicVersion="ASWGmusicVersion", isrcId="ASWGisrcId",
  tempo="ASWGtempo", timeSig="ASWGtimeSig",
  inKey="ASWGinKey", billingCode="ASWGbillingCode"
}
for k,v in pairs(aswgMap) do map["ASWG:" .. k] = v end

--------------------------------------------------------
-- Inject
--------------------------------------------------------
reaper.Undo_BeginBlock()
for metaKey, fieldName in pairs(map) do
  setMeta(metaKey, fieldName, vals[fieldName])
end
reaper.Undo_EndBlock("UCS Inject Render Metadata", -1)

reaper.ShowMessageBox(
  "Injected UCS / ASWG metadata keys (" ..
  (direct_values and "literal values" or "marker references") .. ").",
  "UCS Inject Metadata", 0
)

