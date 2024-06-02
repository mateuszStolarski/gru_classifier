from dataclasses import fields


def class_from_args(className, argDict):
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {k: v for k, v in argDict.items() if k in fieldSet}
    return className(**filteredArgDict)
