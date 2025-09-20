import base64
import fitz
from typing import Any, List, Tuple, Optional, cast
import pandas as pd
import os
import re
import unicodedata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from clientservices.models import ExtarctQaResponseModel, ExtractTextFromYtResponseModel
from clientservices.implementations import (
    DocUtilsImpl,
    ChunkUtilsImpl,
    YoutubeUtilsImpl,
)
from youtube_transcript_api import YouTubeTranscriptApi


class DocUtils(DocUtilsImpl):

    def ExtractTextFromCsv(self, docPath: str) -> str:
        df = cast(Any, pd).read_csv(docPath, header=None)
        allText: list[str] = []

        for row in df.itertuples(index=False):
            for col_index, value in enumerate(row):
                if cast(Any, pd).isna(value):
                    value = None
                else:
                    text = str(value).strip()
                    value = text if text else "None"
                allText.append(f"<<C{col_index+1}-START>>{value}<<C{col_index+1}-END>>")

        return "\n".join(allText)

    def ExtractTextAndImagesFromPdf(self, docPath: str) -> Tuple[str, List[str]]:
        doc: Any = fitz.open(docPath)
        imagesB64: List[str] = []
        imageCounter: int = 1
        finalTextParts: List[str] = []

        for _, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            pageItems: List[Tuple[str, float, str]] = []

            for block in blocks:
                if block["type"] == 0:  # text block
                    for line in block.get("lines", []):
                        y0 = line["bbox"][1]
                        lineText = " ".join(
                            span.get("text", "")
                            for span in line.get("spans", [])
                            if "text" in span
                        ).strip()
                        if lineText:
                            pageItems.append(("text", y0, lineText))
                elif block["type"] == 1:  # image block
                    y0 = block["bbox"][1]
                    imgId = f"image-{imageCounter}"
                    placeholder = f"<<{imgId}>>"

                    xref: Optional[int] = block.get("image") or block.get("number")
                    baseImage = None
                    if isinstance(xref, int) and xref > 0:
                        try:
                            baseImage = doc.extract_image(xref)
                        except Exception:
                            baseImage = None

                    if baseImage and baseImage.get("image"):
                        data = baseImage["image"]
                    else:
                        try:
                            rect = fitz.Rect(block["bbox"])
                            pix = page.get_pixmap(
                                matrix=fitz.Matrix(2, 2), clip=rect, alpha=False
                            )
                            data = pix.tobytes("png")
                        except Exception:
                            data = b""

                    b64Str = base64.b64encode(data).decode("utf-8") if data else ""
                    imagesB64.append(b64Str)
                    pageItems.append(("image", y0, placeholder))
                    imageCounter += 1

            pageItems.sort(key=lambda x: x[1])
            for itemType, _, content in pageItems:
                if itemType == "text":
                    finalTextParts.append(content)
                else:
                    finalTextParts.append(f"\n{content}\n")

        return "\n".join(finalTextParts), imagesB64

    def ExtractTextFromDoc(self, docPath: str) -> Tuple[str, List[str]]:
        ext = os.path.splitext(docPath)[1].lower()
        if ext == ".pdf":
            return self.ExtractTextAndImagesFromPdf(docPath)
        elif ext == ".csv":
            return self.ExtractTextFromCsv(docPath), []
        else:
            raise ValueError(f"Unsupported file format: {ext}")


class ChunkUtils(ChunkUtilsImpl):

    def __init__(self):
        self.ExtarctTextFromDoc = DocUtils()

    def ExtractChunksFromDoc(
        self, file: str, chunkSize: int, chunkOLSize: int | None = 0
    ) -> Tuple[list[str], list[str]]:
        _PAGE_RE = re.compile(r"\bpage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
        _IMAGE_RE = re.compile(r"\s*(<<IMAGE-\d+>>)\s*", re.IGNORECASE)
        _BULLET_LINE_RE = re.compile(r"^[\s•\-\*\u2022\uf0b7FÞ]+(?=\S)", re.MULTILINE)
        _SOFT_HYPHEN_RE = re.compile(r"\u00AD")
        _HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
        _MULTI_NL_RE = re.compile(r"\n{3,}")
        _WS_NL_RE = re.compile(r"[ \t]+\n")
        _WS_RUN_RE = re.compile(r"[ \t]{2,}")

        def _normalizeText(raw: str) -> str:
            t = unicodedata.normalize("NFKC", raw)
            t = _SOFT_HYPHEN_RE.sub("", t)
            t = _PAGE_RE.sub(" ", t)
            t = _BULLET_LINE_RE.sub("", t)
            t = _HYPHEN_BREAK_RE.sub(r"\1\2", t)
            t = _IMAGE_RE.sub(r" \1 ", t)
            t = _WS_NL_RE.sub("\n", t)
            t = _MULTI_NL_RE.sub("\n\n", t)
            t = _WS_RUN_RE.sub(" ", t)
            t = re.sub(r"\s+", " ", t)
            return t.strip()

        def _mergeTinyChunks(chunks: list[str], minChars: int) -> list[str]:
            merged: list[str] = []
            carry = ""
            for ch in chunks:
                chs = ch.strip()
                if not chs:
                    continue
                if _IMAGE_RE.fullmatch(chs) or len(chs) < minChars:
                    if merged:
                        merged[-1] = (merged[-1].rstrip() + " " + chs).strip()
                    else:
                        carry = (carry + " " + chs).strip()
                else:
                    if carry:
                        chs = (carry + " " + chs).strip()
                        carry = ""
                    merged.append(chs)
            if carry:
                if merged:
                    merged[-1] = (merged[-1].rstrip() + " " + carry).strip()
                else:
                    merged = [carry]
            return merged

        text, images = self.ExtarctTextFromDoc.ExtractTextFromDoc(file)
        text = _normalizeText(text)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunkSize,
            chunk_overlap=chunkOLSize or 0,
            separators=["\n\n", "\n", " "],
            is_separator_regex=False,
            length_function=len,
        )
        chunks = splitter.split_text(text)
        chunks = _mergeTinyChunks(chunks, minChars=max(200, chunkSize // 3))
        return (
            chunks,
            images,
        )

    def ExtarctQaFromText(self, text: str) -> ExtarctQaResponseModel:
        questions = re.findall(r"<<C1-START>>(.*?)<<C1-END>>", text, re.DOTALL)
        answers = re.findall(r"<<C2-START>>(.*?)<<C2-END>>", text, re.DOTALL)
        additionalAnswers = re.findall(r"<<C3-START>>(.*?)<<C3-END>>", text, re.DOTALL)

        combinedAnswer: list[str] = []
        for ans, addAns in zip(answers, additionalAnswers):
            if addAns != "None":
                combinedAnswer.append(f"{ans} Alternative solution is {addAns}")
            else:
                combinedAnswer.append(ans)
        return ExtarctQaResponseModel(questions=questions, answers=combinedAnswer)


ytApi = YouTubeTranscriptApi()


class YoutubeUtils(YoutubeUtilsImpl):
    def ExtractText(
        self, videoId: str, chunkSec: int = 30
    ) -> list[ExtractTextFromYtResponseModel]:
        ytApiData = ytApi.fetch(videoId, languages=["hi", "en"])
        chunkResponse: list[ExtractTextFromYtResponseModel] = []
        currentChunkText = []
        currentChunkStart = None

        for item in ytApiData.snippets:
            windowIndex = int(item.start) // chunkSec
            windowStart = windowIndex * chunkSec

            if currentChunkStart is None:
                currentChunkStart = windowStart

            if windowStart == currentChunkStart:
                currentChunkText.append(item.text)
            else:
                chunkResponse.append(
                    ExtractTextFromYtResponseModel(
                        videoId=videoId,
                        chunkText=" ".join(currentChunkText).strip(),
                        chunkUrl=f"https://www.youtube.com/watch?v={videoId}&t={int(currentChunkStart)}s",
                    )
                )
                currentChunkStart = windowStart
                currentChunkText = [item.text]

        if currentChunkText and currentChunkStart is not None:
            chunkResponse.append(
                ExtractTextFromYtResponseModel(
                    videoId=videoId,
                    chunkText=" ".join(currentChunkText).strip(),
                    chunkUrl=f"https://www.youtube.com/watch?v={videoId}&t={int(currentChunkStart)}s",
                )
            )

        return chunkResponse
